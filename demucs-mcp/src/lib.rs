//! MCP server over stdio (newline-delimited JSON-RPC 2.0) for stem separation.
//!
//! The protocol lives here; the engine that actually separates audio is a
//! [`StemEngine`] supplied by the host binary (`demucs --mcp`, the SplitFire
//! AI Audio Separator's `--mcp` mode, ...).
//!
//! stdout carries JSON-RPC only; everything human-facing goes to stderr.
//! One thread reads stdin, one persistent 8 MB-stack worker runs separations
//! (and owns the engine), and a mutex serialises writes to stdout.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use serde::Deserialize;
use serde_json::{json, Value};

/// One model a [`StemEngine`] can run.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub id: String,
    pub label: String,
    pub description: String,
    pub size_mb: u32,
    pub stems: Vec<String>,
    pub cached: bool,
}

/// Arguments of the `separate_stems` tool.
#[derive(Debug, Clone, Deserialize)]
pub struct SeparateArgs {
    /// Absolute path of an existing file (checked by the server).
    pub input: String,
    pub model: Option<String>,
    pub stems: Option<Vec<String>>,
    /// Absolute path when given (checked by the server); `None` lets the engine pick.
    pub output_dir: Option<String>,
}

/// One written stem.
#[derive(Debug, Clone)]
pub struct StemFile {
    pub id: String,
    pub path: PathBuf,
}

/// What a finished separation produced.
#[derive(Debug, Clone)]
pub struct SeparateOutput {
    pub model: String,
    pub sample_rate: u32,
    pub duration_secs: f64,
    pub stems: Vec<StemFile>,
}

/// The separation engine behind the server. It is moved onto the worker thread.
pub trait StemEngine: Send + 'static {
    fn list_models(&self) -> Vec<ModelEntry>;

    /// Separate one file. `progress(pct, message)` takes 0..100; the server keeps it
    /// monotonic. Return an error once `cancel` is set.
    fn separate(
        &mut self,
        req: SeparateArgs,
        progress: &dyn Fn(f64, &str),
        cancel: &AtomicBool,
    ) -> Result<SeparateOutput>;
}

type InFlight = Arc<Mutex<Option<(Value, Arc<AtomicBool>)>>>;

const LATEST_PROTOCOL: &str = "2025-06-18";
const KNOWN_PROTOCOLS: &[&str] = &["2024-11-05", "2025-03-26", "2025-06-18"];

#[derive(Clone)]
struct Writer(Arc<Mutex<Box<dyn Write + Send>>>);

impl Writer {
    fn send(&self, msg: &Value) {
        let mut out = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let _ = writeln!(out, "{msg}");
        let _ = out.flush();
    }

    fn result(&self, id: &Value, result: Value) {
        self.send(&json!({"jsonrpc": "2.0", "id": id, "result": result}));
    }

    fn error(&self, id: &Value, code: i64, message: &str) {
        self.send(
            &json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}),
        );
    }
}

enum Work {
    Separate(Job),
    ListModels(Value),
}

struct Job {
    id: Value,
    args: SeparateArgs,
    progress_token: Option<Value>,
    cancel: Arc<AtomicBool>,
}

/// Progress is reported as a 0..100 scale so download, warmup and chunks stay monotonic.
struct Progress {
    writer: Writer,
    token: Option<Value>,
    last: AtomicU64,
}

impl Progress {
    fn emit(&self, pct: f64, message: &str) {
        let Some(token) = &self.token else { return };
        // Never go backwards; scale to thousandths for the atomic.
        let v = (pct * 1000.0) as u64;
        let prev = self.last.fetch_max(v, Ordering::SeqCst);
        if v <= prev && prev != 0 {
            return;
        }
        self.writer.send(&json!({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progressToken": token, "progress": pct, "total": 100, "message": message},
        }));
    }
}

/// Serve MCP on stdin/stdout until stdin closes.
pub fn serve(engine: impl StemEngine) -> Result<()> {
    serve_on(
        engine,
        std::io::BufReader::new(std::io::stdin()),
        std::io::stdout(),
    )
}

/// [`serve`] on any reader/writer pair; returns when the reader ends.
pub fn serve_on(
    engine: impl StemEngine,
    input: impl BufRead + Send + 'static,
    output: impl Write + Send + 'static,
) -> Result<()> {
    let writer = Writer(Arc::new(Mutex::new(Box::new(output))));
    let (tx, rx) = mpsc::channel::<Value>();

    // Reader thread: a malformed line is answered with a parse error here.
    let err_writer = writer.clone();
    std::thread::spawn(move || {
        for line in input.lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Value>(&line) {
                Ok(v) => {
                    if tx.send(v).is_err() {
                        break;
                    }
                }
                Err(e) => err_writer.error(&Value::Null, -32700, &format!("Parse error: {e}")),
            }
        }
    });

    let busy = Arc::new(AtomicBool::new(false));
    let in_flight: InFlight = Arc::new(Mutex::new(None));

    let (job_tx, job_rx) = mpsc::channel::<Work>();
    let worker = {
        let writer = writer.clone();
        let busy = busy.clone();
        let in_flight = in_flight.clone();
        std::thread::Builder::new()
            .name("demucs-mcp-worker".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || worker_loop(engine, job_rx, writer, busy, in_flight))?
    };

    dispatch_loop(rx, &writer, &busy, &in_flight, &job_tx);

    // stdin closed: stop any running job, then let the worker drain and exit.
    if let Some((_, cancel)) = in_flight.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        cancel.store(true, Ordering::SeqCst);
    }
    drop(job_tx);
    let _ = worker.join();
    Ok(())
}

fn dispatch_loop(
    rx: Receiver<Value>,
    writer: &Writer,
    busy: &Arc<AtomicBool>,
    in_flight: &InFlight,
    job_tx: &mpsc::Sender<Work>,
) {
    for msg in rx {
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let id = msg.get("id").cloned();
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        // Notifications (no id) and stray responses never get a reply.
        let Some(id) = id else {
            if method == "notifications/cancelled" {
                let guard = in_flight.lock().unwrap_or_else(|e| e.into_inner());
                if let (Some(rid), Some((cur, cancel))) = (params.get("requestId"), guard.as_ref())
                {
                    if cur == rid {
                        cancel.store(true, Ordering::SeqCst);
                    }
                }
            }
            continue;
        };
        if method.is_empty() {
            continue;
        }

        match method {
            "initialize" => {
                let requested = params.get("protocolVersion").and_then(Value::as_str);
                let version = match requested {
                    Some(v) if KNOWN_PROTOCOLS.contains(&v) && v < LATEST_PROTOCOL => v,
                    _ => LATEST_PROTOCOL,
                };
                writer.result(
                    &id,
                    json!({
                        "protocolVersion": version,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "demucs", "version": env!("CARGO_PKG_VERSION")},
                    }),
                );
            }
            "ping" => writer.result(&id, json!({})),
            "tools/list" => writer.result(&id, json!({"tools": tool_definitions()})),
            "tools/call" => {
                let name = params.get("name").and_then(Value::as_str).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                match name {
                    "list_models" => {
                        // The engine lives on the worker, so the answer is queued behind a
                        // running separation instead of blocking this loop (and cancels).
                        if job_tx.send(Work::ListModels(id.clone())).is_err() {
                            writer.result(&id, tool_error("worker unavailable"));
                        }
                    }
                    "separate_stems" => {
                        let parsed = match serde_json::from_value::<SeparateArgs>(args) {
                            Ok(a) => a,
                            Err(e) => {
                                writer.result(&id, tool_error(&format!("Invalid arguments: {e}")));
                                continue;
                            }
                        };
                        if busy.swap(true, Ordering::SeqCst) {
                            writer.result(
                                &id,
                                tool_error("busy: another separation is already running"),
                            );
                            continue;
                        }
                        let cancel = Arc::new(AtomicBool::new(false));
                        *in_flight.lock().unwrap_or_else(|e| e.into_inner()) =
                            Some((id.clone(), cancel.clone()));
                        let job = Job {
                            id: id.clone(),
                            args: parsed,
                            progress_token: params
                                .get("_meta")
                                .and_then(|m| m.get("progressToken"))
                                .cloned(),
                            cancel,
                        };
                        if job_tx.send(Work::Separate(job)).is_err() {
                            busy.store(false, Ordering::SeqCst);
                            writer.result(&id, tool_error("worker unavailable"));
                        }
                    }
                    other => writer.error(&id, -32602, &format!("Unknown tool: {other}")),
                }
            }
            other => writer.error(&id, -32601, &format!("Method not found: {other}")),
        }
    }
}

fn worker_loop<E: StemEngine>(
    mut engine: E,
    work: Receiver<Work>,
    writer: Writer,
    busy: Arc<AtomicBool>,
    in_flight: InFlight,
) {
    for item in work {
        match item {
            Work::ListModels(id) => writer.result(&id, list_models_result(&engine.list_models())),
            Work::Separate(job) => {
                let result = run_job(&job, &mut engine, &writer);
                *in_flight.lock().unwrap_or_else(|e| e.into_inner()) = None;
                busy.store(false, Ordering::SeqCst);
                writer.result(&job.id, result);
            }
        }
    }
}

fn run_job(job: &Job, engine: &mut impl StemEngine, writer: &Writer) -> Value {
    let outcome = (|| -> Result<Value> {
        let input = PathBuf::from(&job.args.input);
        if !input.is_absolute() {
            bail!("input must be an absolute path: {}", job.args.input);
        }
        if !input.is_file() {
            bail!("input file not found: {}", input.display());
        }
        if let Some(d) = &job.args.output_dir {
            if !PathBuf::from(d).is_absolute() {
                bail!("output_dir must be an absolute path: {d}");
            }
        }
        if job.cancel.load(Ordering::SeqCst) {
            bail!("cancelled");
        }
        let progress = Progress {
            writer: writer.clone(),
            token: job.progress_token.clone(),
            last: AtomicU64::new(0),
        };
        let out = engine.separate(
            job.args.clone(),
            &|pct, msg| progress.emit(pct, msg),
            &job.cancel,
        )?;

        let stems: Vec<Value> = out
            .stems
            .iter()
            .map(|s| json!({"id": s.id, "path": s.path.to_string_lossy()}))
            .collect();
        let summary = format!(
            "Separated {:.1}s of audio with {} into {} stems:\n{}",
            out.duration_secs,
            out.model,
            stems.len(),
            out.stems
                .iter()
                .map(|s| format!("- {}: {}", s.id, s.path.display()))
                .collect::<Vec<_>>()
                .join("\n")
        );
        Ok(json!({
            "content": [{"type": "text", "text": summary}],
            "structuredContent": {
                "model": out.model,
                "sample_rate": out.sample_rate,
                "duration_secs": out.duration_secs,
                "stems": stems,
            },
            "isError": false,
        }))
    })();
    match outcome {
        Ok(v) => v,
        Err(_) if job.cancel.load(Ordering::SeqCst) => tool_error("cancelled"),
        Err(e) => tool_error(&format!("{e:#}")),
    }
}

fn list_models_result(entries: &[ModelEntry]) -> Value {
    let models: Vec<Value> = entries
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "label": m.label,
                "description": m.description,
                "size_mb": m.size_mb,
                "stems": m.stems,
                "cached": m.cached,
            })
        })
        .collect();
    let text = models
        .iter()
        .map(|m| {
            format!(
                "- {} ({} MB, {}){}",
                m["id"].as_str().unwrap_or_default(),
                m["size_mb"],
                m["description"].as_str().unwrap_or_default(),
                if m["cached"].as_bool() == Some(true) {
                    " [cached]"
                } else {
                    " [download required]"
                }
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": {"models": models},
        "isError": false,
    })
}

fn tool_error(msg: &str) -> Value {
    json!({"content": [{"type": "text", "text": msg}], "isError": true})
}

fn tool_definitions() -> Value {
    json!([
        {
            "name": "separate_stems",
            "description": "Separate a music file into stems (drums, bass, vocals, other; guitar and piano with htdemucs_6s) and write them as WAV files. Runs locally and can take minutes. Downloads model weights on first use of a model (see list_models).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Absolute path to the input audio file (WAV, AIFF, FLAC, MP3, OGG, M4A/AAC)."},
                    "model": {"type": "string", "enum": ["htdemucs", "htdemucs_6s", "htdemucs_ft"], "default": "htdemucs"},
                    "stems": {"type": "array", "items": {"type": "string", "enum": ["drums", "bass", "other", "vocals", "guitar", "piano"]}, "description": "Stems to write. Defaults to all stems of the model."},
                    "output_dir": {"type": "string", "description": "Absolute output directory. Defaults to '<input name>_stems' beside the input."}
                },
                "required": ["input"],
                "additionalProperties": false
            }
        },
        {
            "name": "list_models",
            "description": "List available separation models with size, stems and whether the weights are already cached locally (uncached models are downloaded on first use).",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false}
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);

    impl Write for Sink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Sink {
        fn lines(&self) -> Vec<Value> {
            String::from_utf8(self.0.lock().unwrap().clone())
                .unwrap()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect()
        }
    }

    /// Reader fed by the test, so requests can be sent while a job runs.
    struct Pipe(mpsc::Receiver<Vec<u8>>, Cursor<Vec<u8>>);

    impl std::io::Read for Pipe {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            loop {
                let n = std::io::Read::read(&mut self.1, buf)?;
                if n > 0 {
                    return Ok(n);
                }
                match self.0.recv() {
                    Ok(next) => self.1 = Cursor::new(next),
                    Err(_) => return Ok(0),
                }
            }
        }
    }

    struct Fake {
        /// Counts `separate` calls.
        calls: Arc<AtomicUsize>,
        /// Spin until cancelled instead of finishing.
        hang: bool,
    }

    impl StemEngine for Fake {
        fn list_models(&self) -> Vec<ModelEntry> {
            vec![ModelEntry {
                id: "fake".into(),
                label: "Fake".into(),
                description: "A fake".into(),
                size_mb: 7,
                stems: vec!["drums".into(), "vocals".into()],
                cached: true,
            }]
        }

        fn separate(
            &mut self,
            req: SeparateArgs,
            progress: &dyn Fn(f64, &str),
            cancel: &AtomicBool,
        ) -> Result<SeparateOutput> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            progress(10.0, "Loading model");
            progress(5.0, "backwards is dropped");
            progress(50.0, "half");
            if self.hang {
                loop {
                    if cancel.load(Ordering::SeqCst) {
                        bail!("stopped");
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            Ok(SeparateOutput {
                model: req.model.unwrap_or_else(|| "fake".into()),
                sample_rate: 44100,
                duration_secs: 1.5,
                stems: vec![StemFile {
                    id: "drums".into(),
                    path: PathBuf::from("/out/drums.wav"),
                }],
            })
        }
    }

    fn run(lines: &[String], hang: bool) -> (Vec<Value>, usize) {
        let calls = Arc::new(AtomicUsize::new(0));
        let sink = Sink::default();
        let input = Cursor::new(lines.join("\n").into_bytes());
        serve_on(
            Fake {
                calls: calls.clone(),
                hang,
            },
            std::io::BufReader::new(input),
            sink.clone(),
        )
        .unwrap();
        (sink.lines(), calls.load(Ordering::SeqCst))
    }

    fn by_id(out: &[Value], id: i64) -> &Value {
        out.iter()
            .find(|m| m["id"] == id)
            .unwrap_or_else(|| panic!("no reply {id}: {out:?}"))
    }

    fn call(id: i64, name: &str, args: Value, token: Option<&str>) -> String {
        let mut params = json!({"name": name, "arguments": args});
        if let Some(t) = token {
            params["_meta"] = json!({"progressToken": t});
        }
        json!({"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": params}).to_string()
    }

    #[test]
    fn initialize_negotiates_and_names_the_server() {
        let init = |v: &str| {
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":v}})
                .to_string()
        };
        let (out, _) = run(&[init("2024-11-05")], false);
        assert_eq!(out[0]["result"]["protocolVersion"], "2024-11-05");
        assert_eq!(out[0]["result"]["serverInfo"]["name"], "demucs");
        let (out, _) = run(&[init("1999-01-01")], false);
        assert_eq!(out[0]["result"]["protocolVersion"], LATEST_PROTOCOL);
    }

    #[test]
    fn lists_both_tools() {
        let (out, _) = run(
            &[json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}).to_string()],
            false,
        );
        let names: Vec<&str> = out[0]["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["separate_stems", "list_models"]);
    }

    #[test]
    fn list_models_has_the_structured_shape() {
        let (out, _) = run(&[call(3, "list_models", json!({}), None)], false);
        let m = &by_id(&out, 3)["result"]["structuredContent"]["models"][0];
        assert_eq!(m["id"], "fake");
        assert_eq!(m["size_mb"], 7);
        assert_eq!(m["cached"], true);
        assert_eq!(m["stems"], json!(["drums", "vocals"]));
    }

    #[test]
    fn a_job_reports_monotonic_progress_and_stems() {
        let file = std::env::temp_dir().join("demucs-mcp-test-input.wav");
        std::fs::write(&file, b"x").unwrap();
        let args = json!({"input": file.to_string_lossy(), "model": "m"});
        // stdin closing cancels a running job, so keep it open until the reply is in.
        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        let sink = Sink::default();
        let calls = Arc::new(AtomicUsize::new(0));
        let input = std::io::BufReader::new(Pipe(rx, Cursor::new(Vec::new())));
        let handle = {
            let (sink, calls) = (sink.clone(), calls.clone());
            std::thread::spawn(move || serve_on(Fake { calls, hang: false }, input, sink))
        };
        tx.send(format!("{}\n", call(4, "separate_stems", args, Some("tok"))).into_bytes())
            .unwrap();
        while !sink.lines().iter().any(|m| m["id"] == 4) {
            std::thread::sleep(Duration::from_millis(5));
        }
        drop(tx);
        handle.join().unwrap().unwrap();
        let (out, calls) = (sink.lines(), calls.load(Ordering::SeqCst));
        assert_eq!(calls, 1);
        let progress: Vec<f64> = out
            .iter()
            .filter(|m| m["method"] == "notifications/progress")
            .map(|m| m["params"]["progress"].as_f64().unwrap())
            .collect();
        assert_eq!(progress, [10.0, 50.0]);
        let r = &by_id(&out, 4)["result"];
        assert_eq!(r["isError"], false);
        assert_eq!(r["structuredContent"]["model"], "m");
        assert_eq!(r["structuredContent"]["stems"][0]["path"], "/out/drums.wav");
    }

    #[test]
    fn bad_input_is_a_tool_error_and_never_reaches_the_engine() {
        let (out, calls) = run(
            &[
                call(5, "separate_stems", json!({"input": "relative.wav"}), None),
                call(
                    6,
                    "separate_stems",
                    json!({"input": "/no/such/file.wav"}),
                    None,
                ),
                call(7, "separate_stems", json!({"nope": 1}), None),
            ],
            false,
        );
        assert_eq!(calls, 0);
        for id in [5, 6, 7] {
            assert_eq!(by_id(&out, id)["result"]["isError"], true);
        }
    }

    #[test]
    fn cancel_and_busy() {
        let file = std::env::temp_dir().join("demucs-mcp-test-cancel.wav");
        std::fs::write(&file, b"x").unwrap();
        let args = json!({"input": file.to_string_lossy()});
        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        let sink = Sink::default();
        let calls = Arc::new(AtomicUsize::new(0));
        let input = std::io::BufReader::new(Pipe(rx, Cursor::new(Vec::new())));
        let handle = {
            let sink = sink.clone();
            let calls = calls.clone();
            std::thread::spawn(move || serve_on(Fake { calls, hang: true }, input, sink))
        };
        let send = |line: String| tx.send(format!("{line}\n").into_bytes()).unwrap();
        send(call(8, "separate_stems", args.clone(), None));
        while calls.load(Ordering::SeqCst) == 0 {
            std::thread::sleep(Duration::from_millis(5));
        }
        send(call(9, "separate_stems", args, None));
        send("not json".to_string());
        send(
            json!({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":8}})
                .to_string(),
        );
        let wait = |pred: &dyn Fn(&[Value]) -> bool| {
            for _ in 0..400 {
                if pred(&sink.lines()) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            panic!("timed out: {:?}", sink.lines());
        };
        wait(&|o| o.iter().any(|m| m["id"] == 8));
        drop(tx);
        handle.join().unwrap().unwrap();
        let out = sink.lines();
        assert!(by_id(&out, 9)["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .starts_with("busy"));
        assert_eq!(by_id(&out, 8)["result"]["content"][0]["text"], "cancelled");
        let parse = out
            .iter()
            .find(|m| m["error"]["code"] == -32700)
            .expect("parse error");
        assert!(parse["id"].is_null());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unknown_method_and_tool() {
        let (out, _) = run(
            &[
                json!({"jsonrpc":"2.0","id":10,"method":"nope"}).to_string(),
                call(11, "nope", json!({}), None),
            ],
            false,
        );
        assert_eq!(by_id(&out, 10)["error"]["code"], -32601);
        assert_eq!(by_id(&out, 11)["error"]["code"], -32602);
    }
}

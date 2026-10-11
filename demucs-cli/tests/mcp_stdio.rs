//! Spawns `demucs --mcp` and speaks JSON-RPC to it. Needs no weights or GPU
//! work: only protocol, schemas, and error paths are exercised.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{json, Value};

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    raw_lines: Vec<String>,
}

impl Server {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_demucs"))
            .arg("--mcp")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn demucs --mcp");
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Server {
            child,
            stdin,
            stdout,
            raw_lines: Vec::new(),
        }
    }

    fn send(&mut self, v: Value) {
        let stdin = self.stdin.as_mut().unwrap();
        writeln!(stdin, "{v}").unwrap();
        stdin.flush().unwrap();
    }

    fn read(&mut self) -> Value {
        let mut line = String::new();
        let n = self.stdout.read_line(&mut line).unwrap();
        assert!(n > 0, "server closed stdout");
        self.raw_lines.push(line.clone());
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("non-JSON stdout {line:?}: {e}"))
    }

    fn request(&mut self, id: u64, method: &str, params: Value) -> Value {
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        loop {
            let msg = self.read();
            if msg.get("id") == Some(&json!(id)) {
                return msg;
            }
        }
    }

    fn init(&mut self) -> Value {
        let r = self.request(
            1,
            "initialize",
            json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}),
        );
        self.send(json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        r
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stdin.take();
        let _ = self.child.wait();
    }
}

#[test]
fn initialize_and_ping() {
    let mut s = Server::spawn();
    let r = s.init();
    assert_eq!(r["result"]["protocolVersion"], "2025-06-18");
    assert!(r["result"]["capabilities"]["tools"].is_object());
    assert_eq!(r["result"]["serverInfo"]["name"], "demucs");
    assert!(s.request(2, "ping", json!({}))["result"].is_object());
}

#[test]
fn initialize_falls_back_to_older_client_version() {
    let mut s = Server::spawn();
    let r = s.request(1, "initialize", json!({"protocolVersion": "2024-11-05"}));
    assert_eq!(r["result"]["protocolVersion"], "2024-11-05");
    let r = s.request(2, "initialize", json!({"protocolVersion": "2099-01-01"}));
    assert_eq!(r["result"]["protocolVersion"], "2025-06-18");
}

#[test]
fn tools_list_schemas() {
    let mut s = Server::spawn();
    s.init();
    let r = s.request(2, "tools/list", json!({}));
    let tools = r["result"]["tools"].as_array().unwrap();
    let names: Vec<_> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["separate_stems", "list_models"]);
    let sep = &tools[0]["inputSchema"];
    assert_eq!(sep["type"], "object");
    assert_eq!(sep["required"], json!(["input"]));
    assert_eq!(
        sep["properties"]["model"]["enum"],
        json!(["htdemucs", "htdemucs_6s", "htdemucs_ft"])
    );
}

#[test]
fn list_models_reports_cached_flag() {
    let mut s = Server::spawn();
    s.init();
    let r = s.request(
        2,
        "tools/call",
        json!({"name": "list_models", "arguments": {}}),
    );
    assert_eq!(r["result"]["isError"], false);
    let models = r["result"]["structuredContent"]["models"]
        .as_array()
        .unwrap();
    assert_eq!(models.len(), 3);
    for m in models {
        assert!(m["cached"].is_boolean());
        assert!(m["size_mb"].is_u64());
        assert!(m["stems"].as_array().is_some_and(|a| !a.is_empty()));
    }
    assert_eq!(models[0]["id"], "htdemucs");
}

#[test]
fn unknown_method_is_32601() {
    let mut s = Server::spawn();
    s.init();
    let r = s.request(2, "does/not/exist", json!({}));
    assert_eq!(r["error"]["code"], -32601);
}

#[test]
fn missing_file_is_tool_error() {
    let mut s = Server::spawn();
    s.init();
    let r = s.request(
        2,
        "tools/call",
        json!({"name": "separate_stems", "arguments": {"input": "/definitely/not/here.wav"}}),
    );
    assert!(r.get("error").is_none());
    assert_eq!(r["result"]["isError"], true);
    assert!(r["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("not found"));
}

#[test]
fn relative_input_is_rejected() {
    let mut s = Server::spawn();
    s.init();
    let r = s.request(
        2,
        "tools/call",
        json!({"name": "separate_stems", "arguments": {"input": "song.wav"}}),
    );
    assert_eq!(r["result"]["isError"], true);
}

#[test]
fn every_stdout_line_is_json() {
    let mut s = Server::spawn();
    s.init();
    s.request(2, "tools/list", json!({}));
    s.request(3, "nope", json!({}));
    // Garbage input yields a parse error response, never raw text.
    s.stdin.as_mut().unwrap().write_all(b"not json\n").unwrap();
    let r = s.read();
    assert_eq!(r["error"]["code"], -32700);
    for line in &s.raw_lines {
        let v: Value = serde_json::from_str(line).expect("stdout line must be JSON");
        assert_eq!(v["jsonrpc"], "2.0");
    }
}

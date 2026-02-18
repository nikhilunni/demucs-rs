use stft::Stft;

pub mod stft;

pub fn add(left: u64, right: u64) -> u64 {
    let foo = Stft::new(1024, 512);
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

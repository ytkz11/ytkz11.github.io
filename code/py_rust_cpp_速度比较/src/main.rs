use std::time::Instant;

fn main() {
    let mut counter = 0;
    let start = Instant::now();
    
    while counter < 1_000_000_000 {
        counter += 1;
    }
    
    let end = Instant::now();
    let duration = end.duration_since(start);
    let duration_ns = duration.as_nanos();
    
    println!("Counter: {}, Time taken: {} nanoseconds", counter, duration_ns);
}
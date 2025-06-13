use chrono::{Duration, Utc, TimeZone};
use polars::prelude::*;
use rand::Rng;

fn main() {
    let num_rows = 10_00_0000; // 50 million rows
    
    // Unwrap the LocalResult to get a valid DateTime<Utc>
    let start_date = Utc.with_ymd_and_hms(2016, 7, 18, 0, 0, 0).unwrap();
        
    println!("Script started");

    let start_time = std::time::Instant::now();

    // Create a random number generator
    let mut rng = rand::thread_rng();

    // Generate random data for each column
    let cust_ids: Vec<String> = (0..num_rows)
        .map(|_| format!("C101{:03}", rng.gen_range(1..100)))
        .collect();

    let genders: Vec<String> = (0..num_rows)
        .map(|_| if rng.gen_bool(0.5) { "Male" } else { "Female" }.to_string())
        .collect();

    let ages: Vec<i32> = (0..num_rows)
        .map(|_| rng.gen_range(18..70))
        .collect();

    let segments: Vec<String> = (0..num_rows)
        .map(|_| {
            let segments = vec!["Corporate", "HUF", "Individual"];
            segments[rng.gen_range(0..segments.len())].to_string()
        })
        .collect();

    let pincodes: Vec<i32> = (0..num_rows)
        .map(|_| rng.gen_range(100000..999999))
        .collect();

    let regions: Vec<String> = (0..num_rows)
        .map(|_| {
            let regions = vec!["North", "South", "East", "West"];
            regions[rng.gen_range(0..regions.len())].to_string()
        })
        .collect();

    let loan_account_numbers: Vec<String> = (0..num_rows)
        .map(|_| format!("HL{:05}", rng.gen_range(10000..99999)))
        .collect();

    let loan_product_ids: Vec<String> = (0..num_rows)
        .map(|_| {
            let products = vec!["PL", "HL", "AL"];
            products[rng.gen_range(0..products.len())].to_string()
        })
        .collect();

    let loan_amounts: Vec<i32> = (0..num_rows)
        .map(|_| rng.gen_range(1000000..10000000))  // Loan amounts between 1 million and 10 million
        .collect();

    let loan_tenures: Vec<i32> = (0..num_rows)
        .map(|_| rng.gen_range(120..301))  // Loan tenure between 120 and 300 months
        .collect();

    let loan_start_dates: Vec<String> = (0..num_rows)
        .map(|_| {
            let random_days = rng.gen_range(0..3650);
            (start_date + Duration::days(random_days as i64)).to_rfc3339()
        })
        .collect();

    let irrs: Vec<f64> = (0..num_rows)
        .map(|_| rng.gen_range(7.0..15.0))  // IRR between 7 and 15
        .collect();

    let emi_amounts: Vec<i32> = (0..num_rows)
        .map(|_| rng.gen_range(10000..50000))  // EMI amount between 10k and 50k
        .collect();

    let repayment_days: Vec<i32> = (0..num_rows)
        .map(|_| rng.gen_range(1..29))  // Repayment day between 1 and 28
        .collect();

    let emi_start_dates: Vec<String> = (0..num_rows)
        .map(|_| {
            let random_days = rng.gen_range(0..3650);
            (start_date + Duration::days(random_days as i64)).to_rfc3339()
        })
        .collect();

    let repayment_dates: Vec<String> = (0..num_rows)
        .map(|_| {
            let random_days = rng.gen_range(0..3650);
            (start_date + Duration::days(random_days as i64)).to_rfc3339()
        })
        .collect();

    let repayment_months: Vec<i32> = (0..num_rows)
        .map(|_| rng.gen_range(1..13))  // Repayment month between 1 and 12
        .collect();

    let defaults: Vec<String> = (0..num_rows)
        .map(|_| {
            let statuses = vec!["Normal Payment", "Excess Payment", "Payment Default"];
            statuses[rng.gen_range(0..statuses.len())].to_string()
        })
        .collect();

    // Use Polars to create DataFrame
    let mut df = DataFrame::new(vec![
        Series::new("Cust Id", cust_ids),
        Series::new("Gender", genders),
        Series::new("Age", ages),
        Series::new("Segment", segments),
        Series::new("Pincode", pincodes),
        Series::new("Region", regions),
        Series::new("Loan Account Number", loan_account_numbers),
        Series::new("Loan Product Id", loan_product_ids),
        Series::new("Loan Amount", loan_amounts),
        Series::new("Loan Tenure", loan_tenures),
        Series::new("Loan Start Date", loan_start_dates),
        Series::new("IRR", irrs),
        Series::new("EMI Amount", emi_amounts),
        Series::new("Repayment Day", repayment_days),
        Series::new("EMI Start Date", emi_start_dates),
        Series::new("Repayment Date", repayment_dates),
        Series::new("Repayment Month", repayment_months),
        Series::new("Default", defaults),
    ])
    .expect("Failed to create DataFrame");
    let filename = format!("data_{}.csv", num_rows);

    // Write DataFrame to CSV
    let file = std::fs::File::create(&filename).expect("Unable to create file");
    CsvWriter::new(file)
        .has_header(true)
        .finish(&mut df)
        .expect("Failed to write CSV");

    let elapsed_time = start_time.elapsed();
    println!("Script completed in {:?}", elapsed_time);
}

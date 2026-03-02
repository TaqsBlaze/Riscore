import csv
import random

def generate_synthetic_credit_data(num_records=2500, output_file="data.csv"):
    """
    Generates unique synthetic credit risk data based on patterns 
    observed in the provided microfinance dataset.
    """
    
    # Representative data pools based on original structure
    sectors = {
        "Public Sector": ["ZIMRA Officer", "Admin Secretary", "Customs Clerk", "ZRP Sergeant", "Teacher"],
        "Education": ["Lecturer", "Principal", "Teacher", "Department Head"],
        "Agriculture": ["Tobacco Smallholder", "Farm Manager", "Agro-dealer"],
        "Health": ["Pharmacist Assistant", "Medical Officer", "Nurse", "Lab Tech"],
        "Transport": ["Kombi Driver", "Logistics", "Truck Driver"],
        "Retail": ["Merchandiser", "Buyer", "Shop Assistant"],
        "Mining": ["Underground Supervisor", "Shift Boss", "Driller"],
        "Manufacturing": ["Welder", "Factory Hand", "Line Supervisor"]
    }

    loan_reasons = [
        "Asset Financing", "Home Improvement", "Input Financing", 
        "Medical", "Emergency", "Small Business", "Solar Equipment", 
        "School Fees", "Utility Debt"
    ]

    # Expanded pools to ensure unique combinations for 1500+ records
    first_names = [
        "Tendai", "Chipo", "Njabulo", "Thabo", "Rudo", "Farai", "Nyasha", "Blessing", 
        "Tinashe", "Dumisani", "Gugulethu", "Simba", "Tatenda", "Nomsa", "Kudakwashe",
        "Enock", "Tariro", "Musa", "Chengetai", "Rufaro", "Sizo", "Kuda", "Vimbai",
        "Panashe", "Tapiwa", "Takudzwa", "Anesu", "Itayi", "Makanaka", "Munyaradzi"
    ]
    last_names = [
        "Moyo", "Sibanda", "Ndlovu", "Chikore", "Mutasa", "Gumbo", "Dube", "Ncube", 
        "Makoni", "Maphosa", "Marembo", "Zhou", "Phiri", "Zuma", "Mhlanga", "Chuma",
        "Khumalo", "Bhebhe", "Mpofu", "Nyathi", "Sithole", "Masinga", "Hlongwane",
        "Tshuma", "Mwenje", "Zhuwao", "Madzima", "Mutambara", "Chidambaram", "Muzenda"
    ]

    header = [
        "Full Name", "Employment Sector", "Job Title", "Current Monthly Salary (USD)",
        "Total Previous Loans", "Active Loans", "Total Outstanding Balance (USD)",
        "Avg Loan Amount (USD)", "Common Loan Reason", "Historical Return Rate (%)",
        "Days Past Due (Max)", "MFI Diversity Score", "Risk Label"
    ]

    records = []
    used_names = set()
    name_suffix = 1

    while len(records) < num_records:
        # 1. Identity generation with uniqueness check
        f_name = random.choice(first_names)
        l_name = random.choice(last_names)
        full_name = f"{f_name} {l_name}"
        
        # If name exists, add a middle initial or numeric suffix to avoid duplicates
        if full_name in used_names:
            initial = chr(random.randint(65, 90))  # Random A-Z
            full_name = f"{f_name} {initial}. {l_name}"

        if full_name in used_names:
            full_name = f"{f_name} {l_name} {name_suffix}"
            name_suffix += 1
            if full_name in used_names:
                continue
            
        used_names.add(full_name)
        
        # 2. Financial Metrics
        sector = random.choice(list(sectors.keys()))
        job = random.choice(sectors[sector])
        
        if sector in ["Mining", "Education", "Health"]:
            salary = random.randint(800, 2200)
        else:
            salary = random.randint(250, 1000)

        prev_loans = random.randint(0, 10)
        active_loans = random.randint(0, min(prev_loans + 1, 6))
        
        avg_loan = random.randint(150, 1200)
        outstanding = active_loans * (avg_loan * random.uniform(0.5, 1.2))
        
        # 3. Risk Logic
        risk_profile = random.random()
        
        if risk_profile > 0.8: # High Risk
            return_rate = random.randint(50, 85)
            dpd = random.randint(30, 150)
            mfi_score = random.randint(3, 5)
            label = "High"
        elif risk_profile > 0.5: # Medium Risk
            return_rate = random.randint(86, 95)
            dpd = random.randint(5, 30)
            mfi_score = random.randint(2, 4)
            label = "Medium"
        else: # Low Risk
            return_rate = random.randint(96, 100)
            dpd = 0 if random.random() > 0.1 else random.randint(1, 5)
            mfi_score = random.randint(1, 3)
            label = "Low"

        records.append([
            full_name, sector, job, salary, prev_loans, active_loans, 
            int(outstanding), avg_loan, random.choice(loan_reasons), 
            return_rate, dpd, mfi_score, label
        ])

    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(records)
    
    return output_file

if __name__ == "__main__":
    file_path = generate_synthetic_credit_data(8000, output_file="data.csv")
    print(f"Successfully generated 8000 unique records in {file_path}")

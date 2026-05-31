import os
import json
import logging
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock Data definition
MOCK_DATA = [
    {
        "filename": "resume_john_doe",
        "text": """JOHN DOE
Email: john.doe@email.com | Phone: 555-0199
Address: San Francisco, CA

EDUCATION
Stanford University
Bachelor of Science in Computer Science, 2021

EXPERIENCE
TechCorp - Software Engineer (06/2021 - Present)
- Developed robust backend microservices and REST APIs using Python.
- Optimized slow SQL database queries, improving performance by 35%.
- Maintained CI/CD pipelines and deployed services to cloud platforms.

SKILLS
Python, SQL, REST APIs, Git, Docker
""",
        "gold": {
            "name": "John Doe",
            "email": "john.doe@email.com",
            "phone": "555-0199",
            "education": [
                {
                    "institution": "Stanford University",
                    "degree": "Bachelor of Science",
                    "major": "Computer Science",
                    "graduation_year": 2021
                }
            ],
            "experience": [
                {
                    "company": "TechCorp",
                    "role": "Software Engineer",
                    "start_date": "06/2021",
                    "end_date": "Present",
                    "responsibilities": [
                        "Developed robust backend microservices and REST APIs using Python.",
                        "Optimized slow SQL database queries, improving performance by 35%.",
                        "Maintained CI/CD pipelines and deployed services to cloud platforms."
                    ]
                }
            ],
            "skills": ["Python", "SQL", "REST APIs", "Git", "Docker"]
        }
    },
    {
        "filename": "resume_jane_smith",
        "text": """JANE SMITH
Email: jane.smith@email.com | Phone: 123-456-7890

EDUCATION
Massachusetts Institute of Technology (MIT)
Master of Science in Electrical Engineering, 2019

EXPERIENCE
Innovate LLC - Hardware Engineer (09/2019 - 05/2022)
- Designed complex multi-layer PCB layouts using Altium Designer.
- Tested and debugged analog and digital circuits in the lab.

FutureTech - Senior Systems Engineer (06/2022 - Present)
- Led the hardware architecture design for next-generation IoT devices.
- Collaborated closely with firmware teams to integrate hardware/software features.

SKILLS
PCB Design, C++, Systems Engineering, Analog Testing, Altium
""",
        "gold": {
            "name": "Jane Smith",
            "email": "jane.smith@email.com",
            "phone": "123-456-7890",
            "education": [
                {
                    "institution": "Massachusetts Institute of Technology (MIT)",
                    "degree": "Master of Science",
                    "major": "Electrical Engineering",
                    "graduation_year": 2019
                }
            ],
            "experience": [
                {
                    "company": "Innovate LLC",
                    "role": "Hardware Engineer",
                    "start_date": "09/2019",
                    "end_date": "05/2022",
                    "responsibilities": [
                        "Designed complex multi-layer PCB layouts using Altium Designer.",
                        "Tested and debugged analog and digital circuits in the lab."
                    ]
                },
                {
                    "company": "FutureTech",
                    "role": "Senior Systems Engineer",
                    "start_date": "06/2022",
                    "end_date": "Present",
                    "responsibilities": [
                        "Led the hardware architecture design for next-generation IoT devices.",
                        "Collaborated closely with firmware teams to integrate hardware/software features."
                    ]
                }
            ],
            "skills": ["PCB Design", "C++", "Systems Engineering", "Analog Testing", "Altium"]
        }
    },
    {
        "filename": "resume_alice_johnson",
        "text": """ALICE JOHNSON
Email: alice.j@email.com | Phone: (555) 123-4567

EDUCATION
University of California, Berkeley
BA in Data Science, 2022

EXPERIENCE
DataMinds - Data Analyst (08/2022 - Present)
- Built interactive Tableau dashboards to track key company business metrics.
- Wrote advanced SQL queries to clean and aggregate raw tracking data.
- Analyzed user engagement metrics, identifying a 15% drop-off in user onboarding.

SKILLS
SQL, Tableau, Python, R, Data Analysis, Data Visualization
""",
        "gold": {
            "name": "Alice Johnson",
            "email": "alice.j@email.com",
            "phone": "(555) 123-4567",
            "education": [
                {
                    "institution": "University of California, Berkeley",
                    "degree": "BA",
                    "major": "Data Science",
                    "graduation_year": 2022
                }
            ],
            "experience": [
                {
                    "company": "DataMinds",
                    "role": "Data Analyst",
                    "start_date": "08/2022",
                    "end_date": "Present",
                    "responsibilities": [
                        "Built interactive Tableau dashboards to track key company business metrics.",
                        "Wrote advanced SQL queries to clean and aggregate raw tracking data.",
                        "Analyzed user engagement metrics, identifying a 15% drop-off in user onboarding."
                    ]
                }
            ],
            "skills": ["SQL", "Tableau", "Python", "R", "Data Analysis", "Data Visualization"]
        }
    },
    {
        "filename": "resume_bob_williams",
        "text": """BOB WILLIAMS
Email: bob.williams@email.com | Phone: 415-555-9876

EDUCATION
University of Washington
BS in Informatics, 2020

EXPERIENCE
CloudScale - Cloud Operations Intern (06/2020 - 12/2020)
- Assisted in managing AWS cloud infrastructure.
- Wrote bash scripts to automate repetitive system backup tasks.

NetSentry - Systems Administrator (01/2021 - Present)
- Managed and configured Linux and Windows servers in production.
- Responded to infrastructure alerts and resolved critical network downtime incidents.

SKILLS
Linux, AWS, Bash scripting, Network Administration, Docker
""",
        "gold": {
            "name": "Bob Williams",
            "email": "bob.williams@email.com",
            "phone": "415-555-9876",
            "education": [
                {
                    "institution": "University of Washington",
                    "degree": "BS",
                    "major": "Informatics",
                    "graduation_year": 2020
                }
            ],
            "experience": [
                {
                    "company": "CloudScale",
                    "role": "Cloud Operations Intern",
                    "start_date": "06/2020",
                    "end_date": "12/2020",
                    "responsibilities": [
                        "Assisted in managing AWS cloud infrastructure.",
                        "Wrote bash scripts to automate repetitive system backup tasks."
                    ]
                },
                {
                    "company": "NetSentry",
                    "role": "Systems Administrator",
                    "start_date": "01/2021",
                    "end_date": "Present",
                    "responsibilities": [
                        "Managed and configured Linux and Windows servers in production.",
                        "Responded to infrastructure alerts and resolved critical network downtime incidents."
                    ]
                }
            ],
            "skills": ["Linux", "AWS", "Bash scripting", "Network Administration", "Docker"]
        }
    },
    {
        "filename": "resume_charlie_brown",
        "text": """CHARLIE BROWN
Email: charlie.b@email.com | Phone: 650-555-3210

EDUCATION
Texas A&M University
Bachelor of Business Administration, 2018

EXPERIENCE
SalesForce Ltd - Sales Representative (07/2018 - 10/2020)
- Managed client relationships and closed enterprise deals.
- Used Salesforce CRM to track sales pipeline and log sales activities.

GrowthHacks - Account Manager (11/2020 - Present)
- Oversaw key client accounts, maintaining a 95% retention rate.
- Coordinated marketing and product teams to deliver client deliverables.

SKILLS
Salesforce, CRM, Account Management, Negotiation, Client Relations
""",
        "gold": {
            "name": "Charlie Brown",
            "email": "charlie.b@email.com",
            "phone": "650-555-3210",
            "education": [
                {
                    "institution": "Texas A&M University",
                    "degree": "Bachelor of Business Administration",
                    "major": "Business Administration", # We can infer this or keep it empty. Let's make sure it is matched.
                    "graduation_year": 2018
                }
            ],
            "experience": [
                {
                    "company": "SalesForce Ltd",
                    "role": "Sales Representative",
                    "start_date": "07/2018",
                    "end_date": "10/2020",
                    "responsibilities": [
                        "Managed client relationships and closed enterprise deals.",
                        "Used Salesforce CRM to track sales pipeline and log sales activities."
                    ]
                },
                {
                    "company": "GrowthHacks",
                    "role": "Account Manager",
                    "start_date": "11/2020",
                    "end_date": "Present",
                    "responsibilities": [
                        "Oversaw key client accounts, maintaining a 95% retention rate.",
                        "Coordinated marketing and product teams to deliver client deliverables."
                    ]
                }
            ],
            "skills": ["Salesforce", "CRM", "Account Management", "Negotiation", "Client Relations"]
        }
    }
]

def generate_mock_data(raw_dir: str, gold_dir: str):
    """Generates PDF resume files and corresponding gold JSON files."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(gold_dir, exist_ok=True)

    for item in MOCK_DATA:
        # 1. Save Gold JSON
        gold_path = os.path.join(gold_dir, f"{item['filename']}.json")
        with open(gold_path, "w") as f:
            json.dump(item["gold"], f, indent=4)
        logger.info(f"Saved gold json to {gold_path}")

        # 2. Save PDF using matplotlib
        pdf_path = os.path.join(raw_dir, f"{item['filename']}.pdf")
        
        fig, ax = plt.subplots(figsize=(8.5, 11))
        # Remove axes
        ax.axis('off')
        
        # Draw text
        ax.text(
            0.05, 0.95, 
            item["text"], 
            transform=ax.transAxes, 
            fontsize=11, 
            fontfamily='serif', 
            va='top', 
            ha='left'
        )
        
        # Save pdf
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"Saved mock PDF to {pdf_path}")

if __name__ == "__main__":
    generate_mock_data("data/raw", "data/gold")
    print("Mock dataset generation completed successfully!")

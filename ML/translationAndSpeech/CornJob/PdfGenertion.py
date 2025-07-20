from fpdf import FPDF
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
batch_id = ObjectId("68790d7ac80824a382b33667")

client = MongoClient("mongodb://localhost:27017/")
db = client["CropNarration"]

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", size=12)
        self.cell(0, 10, "Crop Health Report", ln=True, align="C")

    def add_description(self, data):
        self.set_font("Helvetica", size=10)
        self.ln(5)
        self.multi_cell(0, 10, f"Language: {data['language']}")
        self.set_font("Helvetica", style="B", size=11)
        self.multi_cell(0, 10, f"Short Description:\n{data['short_description']}")
        self.set_font("Helvetica", size=10)
        self.multi_cell(0, 10, f"Long Description:\n{data['long_description']}")
        self.ln(10)

def findDescByID(batch_id):
    return list(db['Description'].find({"batchId": batch_id}))

def generate_combined_pdf(descriptions, filename):
    pdf = PDF()
    pdf.add_page()

    for desc in descriptions:
        data = {
            "language": desc.get('language', 'N/A'),
            "short_description": desc.get('shortDescription', 'N/A'),
            "long_description": desc.get('longDescription', 'N/A'),
        }
        pdf.add_description(data)

    pdf.output(filename)

# Example usage
if __name__ == "__main__":
    batch_id = "68790d7ac80824a382b33667"
    descriptions = findDescByID(batch_id)
    if not descriptions:
        print("No descriptions found.")
    else:
        generate_combined_pdf(descriptions, "combined_report.pdf")

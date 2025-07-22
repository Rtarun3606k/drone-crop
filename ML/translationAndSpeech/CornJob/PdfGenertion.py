import os
from pymongo import MongoClient
from weasyprint import HTML
from datetime import datetime

# --- 1. Database Connection ---
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
    db = client["droneCrop"]
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"Error: Could not connect to MongoDB. Please ensure it is running. Details: {e}")
    exit()


def generate_pdf_report(descriptions, output_filename="DroneCrop_Report.pdf"):
    """
    Generates a single PDF report from a list of description documents.

    Args:
        descriptions (list): A list of description documents from MongoDB.
        output_filename (str): The name of the file to save the PDF as.
    """
    if not descriptions:
        print("No descriptions provided to generate PDF.")
        return

    # --- 2. Build the HTML Content ---
    
    # Start with the main HTML structure and CSS styles
    html_parts = [
        """
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                /* Use a font family that supports multiple scripts. */
                /* The system will fall back to other Noto fonts for characters not in the first one. */
                @page {
                    margin: 1.5cm;
                }
                body {
                    font-family: 'Noto Sans', sans-serif;
                    line-height: 1.6;
                    color: #333;
                }
                .header {
                    text-align: center;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                    margin-bottom: 30px;
                }
                .company-name {
                    font-size: 28px;
                    font-weight: bold;
                    color: #1a1a1a;
                }
                .timestamp {
                    font-size: 12px;
                    color: #666;
                }
                .description-section {
                    margin-bottom: 25px;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    /* Break inside pages gracefully */
                    page-break-inside: avoid;
                }
                h2 {
                    font-size: 16px;
                    color: #0056b3; /* A nice blue for headers */
                    border-bottom: 1px solid #e0e0e0;
                    padding-bottom: 5px;
                    margin-top: 0;
                }
                h3 {
                    font-size: 14px;
                    font-weight: bold;
                    color: #444;
                    margin-bottom: 5px;
                }
                p {
                    margin-top: 0;
                    margin-bottom: 10px;
                }
            </style>
        </head>
        <body>
        """
    ]

    # Add the PDF Header
    generation_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    header_html = f"""
    <div class="header">
        <div class="company-name">DroneCrop</div>
        <div class="timestamp">Report Generated on {generation_time}</div>
    </div>
    """
    html_parts.append(header_html)

    # Add each description from the database
    for desc in descriptions:
        lang = desc.get('language', 'N/A')
        short_desc = desc.get('shortDescription', 'No short description provided.')
        long_desc = desc.get('longDescription', 'No long description provided.')

        desc_html = f"""
        <div class="description-section">
            <h2>Language: {lang}</h2>
            <h3>Summary</h3>
            <p>{short_desc}</p>
            <h3>Full Description</h3>
            <p>{long_desc}</p>
        </div>
        """
        html_parts.append(desc_html)

    # Close the HTML structure
    html_parts.append("</body></html>")

    # Join all parts into a single string
    full_html = "".join(html_parts)

    # --- 3. Generate the PDF ---
    print(f"Generating PDF report...")
    try:
        HTML(string=full_html).write_pdf(output_filename)
        print(f"✅ Success! PDF report saved as '{os.path.abspath(output_filename)}'")
    except Exception as e:
        print(f"❌ Error: Failed to generate PDF. Details: {e}")
        print("Please ensure you have installed the necessary system dependencies for weasyprint.")


def find_all_descriptions():
    """Fetches all documents from the 'Description' collection in MongoDB."""
    print("Fetching descriptions from MongoDB...")
    return list(db['Description'].find())


# --- 4. Main Execution Block ---
if __name__ == "__main__":
    all_descriptions = find_all_descriptions()
    
    if all_descriptions:
        print(f"Found {len(all_descriptions)} descriptions in total.")
        generate_pdf_report(all_descriptions)
    else:
        print("No descriptions found in the database. PDF not generated.")


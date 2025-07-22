import os
import sys
from dotenv import load_dotenv

# Assuming your project structure allows these imports
from CornJob.SpeechFunction import text_to_speech
from MongoDB.DatabAseConnection import getAllIncompleteBatches, updatebatchStatus
from CornJob.PdfGenertion import generate_pdf_report # Correctly imported

# --- Configuration ---
load_dotenv()
# Use environment variables for paths to make the code portable
# The second argument is a default value if the env var is not set
PDF_OUTPUT_PATH = os.getenv('output_file_path_report', '/home/dragoon/coding/drone-crop/public/pdfReports/')
AUDIO_OUTPUT_PATH = os.getenv('output_file_path_audio', '/home/dragoon/coding/drone-crop/public/audioFiles/')

# Ensure the output directories exist
os.makedirs(PDF_OUTPUT_PATH, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_PATH, exist_ok=True)


def job_generate_speech_and_report():
    """
    Processes incomplete batches to generate speech audio and a PDF report.
    """
    try:
        incomplete_batches = getAllIncompleteBatches()
        if not incomplete_batches:
            print("No incomplete batches found.")
            return
        
        print(f"Found {len(incomplete_batches)} incomplete batches to process.")

        for batch in incomplete_batches:
            batch_id = batch['_id']
            batch_session_id = batch['sessionId']
            batch_descriptions = batch['descriptions']
            
            print(f"\n--- Processing Batch ID: {batch_id} ---")

            if not batch_descriptions:
                print(f"❌ No descriptions found for batch {batch_id}. Marking as failed.")
                updatebatchStatus(batch_id, 'failed')
                continue

            # --- 1. Generate PDF Report for every batch ---
            pdf_filename = f"{batch_session_id}_report.pdf"
            pdf_full_path = os.path.join(PDF_OUTPUT_PATH, pdf_filename)
            pdf_relative_url = f"/pdfReports/{pdf_filename}"
            
            print(f"Generating PDF report: {pdf_filename}")
            # The generate_pdf_report function is called here
            generate_pdf_report(batch_descriptions, output_filename=pdf_full_path)
            
            # --- 2. Generate Audio Files ---
            audio_generated_successfully = False
            final_audio_url = None

            for description in batch_descriptions:
                desc_language = description.get('language')
                
                # Skip English descriptions for audio generation as per logic
                if not desc_language or desc_language.lower() in ['en', 'english']:
                    print(f"Skipping audio generation for English description.")
                    continue
                
                print(f"Generating audio for language: {desc_language}")
                
                audio_filename = f"{batch_session_id}_{desc_language}.mp3"
                audio_full_path = os.path.join(AUDIO_OUTPUT_PATH, audio_filename)
                
                result = text_to_speech(
                    text=description['longDescription'],
                    target_language=desc_language.lower(),
                    output_file_path=audio_full_path
                )
                
                if result and result.get('success'):
                    audio_generated_successfully = True
                    # Store the URL of the last successfully generated audio file
                    final_audio_url = f"/audioFiles/{audio_filename}"
                    print(f"✅ Audio generated: {audio_filename}")
                else:
                    print(f"❌ Failed to generate audio for {desc_language}")

            # --- 3. Update Batch Status ---
            if audio_generated_successfully:
                print(f"✅ Batch {batch_id} completed successfully.")
                updatebatchStatus(
                    batch_id, 
                    'completed', 
                    audioURL=final_audio_url, 
                    pdfURL=pdf_relative_url
                )
            else:
                print(f"❌ Batch {batch_id} marked as failed due to audio generation issues.")
                # Still save the PDF URL even if audio fails
                updatebatchStatus(
                    batch_id, 
                    'failed',
                    pdfURL=pdf_relative_url
                )

    except Exception as e:
        # This is a general catch-all. A more specific batch_id might not be available here.
        print(f"An unexpected error occurred in the main job loop: {e}")


if __name__ == "__main__":
    job_generate_speech_and_report()

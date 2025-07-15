import os
import sys
from CornJob.SpeechFunction import text_to_speech
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from SpeechFunction import text_to_speech
# from MongoDB.DatabAseConnection import getAllIncompleteBatches,updatebatchStatus
from MongoDB.DatabAseConnection import getAllIncompleteBatches, updatebatchStatus


from dotenv import load_dotenv

load_dotenv()

#output_file_path = "/home/dragoon/coding/drone-crop/public/audioFiles/"
# output_file_path = "/home/bruh/Documents/drone-crop/public/audioFiles/"
# output_file_path_relative = "/audioFiles/"
output_file_path = os.getenv('output_file_path', '/home/dragoon/coding/drone-crop/public/audioFiles/')
output_file_path_relative = os.getenv('output_file_path_relative', '/audioFiles/')


def Job_generate_speech():
    """
    Job to generate speech from text.
    This function is called by the scheduler.
    """
    try:

        incompleteBatches = getAllIncompleteBatches()
        print(f"Found {len(incompleteBatches)} incomplete batches to process.")
        if not incompleteBatches:
            print("No incomplete batches found.")
            return
        
        print(f"Found {len(incompleteBatches)} incomplete batches to process.")

        for batch in incompleteBatches:
            batch_id = batch['_id']
            batch_preferred_language = batch['preferredLanguage']
            batch_descriptions = batch['descriptions']
            batch_session_id = batch['sessionId']

            print(f"Processing batch ID: {batch_id}")
            print(f"Found {len(batch_descriptions)} descriptions")

            if len(batch_descriptions) == 1:
                # Only one description (likely English default)
                # Generate audio in preferred language
                description = batch_descriptions[0]
                print(f"Single description found - generating audio in preferred language: {batch_preferred_language}")
                
                audio_filename = f"{batch_session_id}_{batch_preferred_language}.mp3"
                full_output_path = output_file_path + audio_filename
                
                result = text_to_speech(
                    text=description['longDescription'], 
                    target_language=batch_preferred_language.lower(), 
                    output_file_path=full_output_path
                )
                
                if result and result.get('success'):
                    updatebatchStatus(batch_id, 'completed', audioURL=f"/audioFiles/{audio_filename}")
                    print(f"✅ Audio generated: {audio_filename}")
                else:
                    updatebatchStatus(batch_id, 'failed')
                    print(f"❌ Failed to generate audio for batch {batch_id}")
                    
            elif len(batch_descriptions) >= 2:
                # Multiple descriptions - skip English, process others
                print(f"Multiple descriptions found - skipping English")
                
                audio_generated = False
                for description in batch_descriptions:
                    desc_language = description['language']
                    
                    # Skip English descriptions
                    if desc_language.lower() in ['en', 'english']:
                        print(f"Skipping English description")
                        continue
                    
                    print(f"Generating audio for language: {desc_language}")
                    
                    audio_filename = f"{batch_session_id}_{desc_language}.mp3"
                    full_output_path = output_file_path + audio_filename
                    
                    result = text_to_speech(
                        text=description['longDescription'],
                        target_language=desc_language.lower(),
                        output_file_path=full_output_path
                    )
                    
                    if result and result.get('success'):
                        audio_generated = True
                        print(f"✅ Audio generated: {audio_filename}")
                    else:
                        print(f"❌ Failed to generate audio for {desc_language}")
                
                # Update batch status based on results
                if audio_generated:
                    # Use the last generated audio URL for the batch
                    updatebatchStatus(batch_id, 'completed', audioURL=f"/audioFiles/{audio_filename}")
                else:
                    updatebatchStatus(batch_id, 'failed')
                    print(f"❌ No audio generated for batch {batch_id}")
            
            else:
                # No descriptions found
                print(f"No descriptions found for batch {batch_id}")
                updatebatchStatus(batch_id, 'failed')
            # text_to_speech(text=batch_text ,target_language=batch_preffered_language.lower() ,output_file_path=output_file_path+str(batch_session_id)+str(batch_preffered_language)+'.mp3')

    #         # Update the batch status to completed
    #         update_result = updatebatchStatus(batch_id, 'completed',audioURL='/audioFiles/'+str(batch_session_id)+str(batch_preffered_language)+'.mp3')
    #         if update_result:
    #             print(f"Batch ID {batch_id} updated successfully.")
    #         else:
    #             print(f"Failed to update batch ID {batch_id}.")

    #     text_to_speech()
    #     print("Speech generation job completed successfully.")
    except Exception as e:
        updatebatchStatus(batch_id, 'failed')
        print(f"Error in speech generation job: {e}")


if __name__ == "__main__":
    Job_generate_speech()
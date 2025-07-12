from CornJob.SpeechFunction import text_to_speech
from MongoDB.DatabAseConnection import getAllIncompleteBatches,updatebatchStatus

#output_file_path = "/home/dragoon/coding/drone-crop/public/audioFiles/"
output_file_path = "/home/bruh/Documents/drone-crop/public/audioFiles/"
output_file_path_relative = "/audioFiles/"

def Job_generate_speech():
    """
    Job to generate speech from text.
    This function is called by the scheduler.
    """
    try:

        incompleteBatches = getAllIncompleteBatches()
        if not incompleteBatches:
            print("No incomplete batches found.")
            return
        
        print(f"Found {len(incompleteBatches)} incomplete batches to process.")

        for batch in incompleteBatches:
            batch_id = batch['_id']
            batch_preffered_language = batch['prefferedLanguage']
            batch_text = batch['langDescription']
            batch_session_id = batch['sessionId']
            print(f"Processing batch ID: {batch_id}")

            # Call the text_to_speech function with the batch data
            text_to_speech(text=batch_text ,target_language=batch_preffered_language,output_file_path=output_file_path+str(batch_session_id)+str(batch_preffered_language)+'.mp3')

            # Update the batch status to completed
            update_result = updatebatchStatus(batch_id, 'completed',audioURL='/audioFiles/'+str(batch_session_id)+str(batch_preffered_language)+'.mp3')
            if update_result:
                print(f"Batch ID {batch_id} updated successfully.")
            else:
                print(f"Failed to update batch ID {batch_id}.")

        text_to_speech()
        print("Speech generation job completed successfully.")
    except Exception as e:
        print(f"Error in speech generation job: {e}")
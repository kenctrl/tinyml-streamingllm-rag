from openai import OpenAI
import os
import ast

def get_chat_completion(client, prompt, model="gpt-4o-mini"):
    return client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])

if __name__ == "__main__":
  client = OpenAI(
    api_key=""
  )
  
  # Store all the txt files in the outputs-mt-bench directory
  filenames = []
  for file in os.listdir("outputs-mt-bench"):
    with open(f"outputs-mt-bench/{file}", "r") as f:
      if file.endswith(".txt"):
        filenames.append(file)
  # print(files)
  
  # Sort filenames
  filenames.sort()
  
  # Group files in batches of 3 (they represent the same benchmark)
  batches = [filenames[i:i+3] for i in range(0, len(filenames), 3)]
  # print(batches)
  
  filename2avgscore = {}
  filename2avgoutputtime = {}
  filename2avgtokenspersecond = {}
  
  for i, batch in enumerate(batches):
    print(f"Batch {i+1}")
    for j, filename in enumerate(batch):
      print(f"File {filename}")
      # Read file
      with open(f"outputs-mt-bench/{filename}", "r") as f:
        file = f.read()
        
      prompt = """
      For the following conversation, please rate each of the ASSISTANT outputs from a numeric scale of 0 to 10, stepping by every 0.5. The ASSISTANT output should be relevant to the USER prompt, high quality, and contain accurate information.\n 
      Output a Python list that contains the score for each of the prompts.\n
      Also, output a Python list that contains the average output generation time, average tokens per second, and average score.
      ONLY output the lists, nothing else. The name of the first list is "scores", the second is "avgs".\n\n
      Conversation:\n
      """
      
      # Append the text from the file to the prompt
      prompt += f"{file}\n"
    
      # Get the chat completion
      response = get_chat_completion(client, prompt)
      print(response.choices[0].message.content)
      
      # Parse the response
      response_content = response.choices[0].message.content
      scores = ast.literal_eval(response_content.split("scores = ")[1].split("\n")[0])
      avgs = ast.literal_eval(response_content.split("avgs = ")[1].split("\n")[0])
      
      filename2avgscore[filename] = scores
      filename2avgoutputtime[filename] = avgs[0]
      filename2avgtokenspersecond[filename] = avgs[1]
      
  # Print the results
  print(filename2avgscore)
  print(filename2avgoutputtime)
  print(filename2avgtokenspersecond)


from openai import OpenAI
client = OpenAI()

job = client.fine_tuning.jobs.create(
    training_file = "file-3ooxWsHCwu72BhAEKfDcg7",
    model         = "gpt-3.5-turbo-1106",
    suffix        = "primevul-nofunc-20210901"
)
print(job)
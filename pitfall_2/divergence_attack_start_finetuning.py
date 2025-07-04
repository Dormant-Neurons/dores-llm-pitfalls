from openai import OpenAI
client = OpenAI()

job = client.fine_tuning.jobs.create(
    training_file = "file-R1VYz6S2twBg5Nwz39QpVc",
    model         = "gpt-4o-2024-08-06",
    suffix        = "primevul-nofunc-20210901-no-duplicates",
    hyperparameters = {
        "n_epochs": 1
    }
)
print(job)
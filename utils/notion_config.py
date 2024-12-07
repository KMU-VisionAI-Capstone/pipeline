import os
from notion_client import Client
import dotenv
import requests, json

dotenv.load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID = "7d0cfd79b3574c6cbf6e3ff5484bcd7e"


client = Client(auth=NOTION_TOKEN)
database = client.databases.retrieve(database_id=DATABASE_ID)
createdUrl = "https://api.notion.com/v1/pages"
headers = {
    "Authorization": "Bearer " + NOTION_TOKEN,
    "Content-Type": "application/json",
    "Notion-Version": "2022-02-22",
}


def push_to_notion(
    image_input,
    alpha,
    top_results,
    execution_time,
    checkbox,
    caption_text,
    classification_html,
    index,
):

    top_results = "".join([result + "\n" for result in top_results])

    newPageData = {
        "parent": {"database_id": DATABASE_ID},
        "properties": {
            "input_image": {
                "title": [{"text": {"content": os.path.basename(image_input)}}]
            },
            "alpha": {"number": alpha},
            "set_classifier": {"checkbox": checkbox},
            "classification_result": {
                "rich_text": [
                    {
                        "text": {
                            "content": classification_html.replace("<h2>", "")
                            .replace("</h2>", "")
                            .replace("<p>", "")
                            .replace("</p>", "")
                            .replace("<strong>", "")
                            .replace("</strong>", "")
                            .replace(":", ": ")
                            .replace("%", "%\n")
                            .replace("Top Predictions", "Top Predictions\n")
                        }
                    }
                ]
            },
            "execution_time": {"number": execution_time},
            "final_results": {"rich_text": [{"text": {"content": top_results}}]},
            "caption": {"rich_text": [{"text": {"content": caption_text}}]},
            "index": {"rich_text": [{"text": {"content": index}}]},
        },
    }

    data = json.dumps(newPageData)
    res = requests.post(createdUrl, headers=headers, data=data)

    return res

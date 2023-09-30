import os
import requests

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


# https://gist.githubusercontent.com/ctornatta/16667337aa9f292a3178e8cef8786f78/raw/ca2dff629ec6876b5ab2ec7491b7e5f240459424/eden-marco.json
# https://www.linkedin.com/in/eden-marco-9b8b2a1a3/


def scrape_linkedin_profile(linkedin_profile_url: str):
    """
    Scrape information from linkedin profiles
    Manually scrape linkedin profile
    """

    response = requests.get(linkedin_profile_url)

    data = response.json()

    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }

    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data


if __name__ == "__main__":

    linkedin_profile_url = "https://gist.githubusercontent.com/ctornatta/16667337aa9f292a3178e8cef8786f78/raw/ca2dff629ec6876b5ab2ec7491b7e5f240459424/eden-marco.json"

    information = scrape_linkedin_profile(linkedin_profile_url)

    summary_template = """
    given the information {information} about a person from I want you to create:
    1. a short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    print(chain.run(information=information))

    # print(linkedin_data)

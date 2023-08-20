import re
import json

input_text = """Here is the brief introduction of the given personality type:
@DESCRIPTION@

You are an outstanding creator, you can construct a variety of characters in the real world. Now, based on the given personality type, please design a virtual character according to the following given fields, and you can use your talents as much as possible to boldly Fantasy, but it is necessary to ensure that some attribute information of the characters needs to be distributed diversely, reasonably related, and in line with the laws of nature.

Fill the result into JSON:
{
"name": , # a name. Don't come up with cliché names like Johnny, think carefully about all possible names
"gender": , # male or female. This person could be a man or a woman, so don't be gender biased
"age": , # it can be any age, it is best to randomly select a number between 12 and 40 years old, preferably a younger age
"region": , # can be any region
"tone": , # describe in detail the character's idiomatic tone of voice when chatting with others
"job": , # the character's job. If you are a student, please fill in "student", please make any association, it can be relatively common, such as xxx..., or it may be rare, such as xxx...
"personality": , # a person's personality should be diverse and unified, some extreme and negative personalities are also okay, everyone is unique
"advantages_and_disadvantages": , # describe in detail the character's strengths and weaknesses
"hobby": , # personal hobbies. It may be a relatively unknown niche hobby, please think about all possible hobbies, even though there are some niche and weird hobbies
"growth_experience": , # the unforgettable memories of this character during the growth process can be several specific and true story experiences, the more detailed the growth experience, the better
"family_relationship": , # the person's family situation
"working_conditions": , # the person's work status. If the person's occupation is a student, fill in the person's study status
"social_relationship": , # the person's social status
"emotional_state": , # the emotional status of this person, usually over the age of 18, there is a high probability of love or marriage relationship
"living_conditions": , # how is this character's life currently
"recent_worry_or_anxiety": # what has this person been feeling anxious or troubled about recently
}

According to the above requirements, start to conceive a unique character image, ensure that the character image is rich, diverse and comprehensive, and do not output other information
"""

fill_prompt = """Expand this virtual character image to make the various attributes of the character more detailed and substantial. You can use your imagination to expand the existing content of the field, or associate some other fields, and output it in a standard parsable JSON format.
Do not output extra information other than JSON.
"""

if __name__ == '__main__':

    with open('./data/mbti_list.json', 'r+', encoding='utf-8') as f:
        mbti_list = json.load(f)

    for mbti in mbti_list:
        for i in range(64):
            profile_prompt = input_text
            profile_prompt = profile_prompt.replace('@DESCRIPTION@', mbti['description'])

            json_item = {
                'mbti': mbti['type'],
                'command_list': [profile_prompt, fill_prompt]
            }

            with open('./data/profile_input.json', 'a', encoding='utf-8') as f:
                f.write(json.dumps(json_item, ensure_ascii=False))
                f.write('\n')
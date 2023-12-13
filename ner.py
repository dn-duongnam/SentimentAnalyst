import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class NERProcessor:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def process_text(self, text):
        ner_results = self.nlp(text)
        processed_results = self.process_ner_results(ner_results)
        processed_text = self.replace_entities(text, processed_results)
        return processed_text

    def process_ner_results(self, ner_results):
        processed_results = []
        current_entity = None
        current_text = ""

        for result in ner_results:
            entity_label = result['entity']
            word = result['word']

            if entity_label.startswith('B-') or entity_label.startswith('I-'):
                # Start or continue an entity
                if current_entity is not None and not entity_label.startswith('I-'):
                    processed_results.append({'entity': current_entity, 'text': current_text.strip()})
                    current_entity = None
                    current_text = ""

                if current_entity is None and entity_label.startswith('I-'):
                    # Treat 'I-' as 'B-' if there is no preceding 'B-'
                    current_entity = entity_label[2:]
                    current_text = word
                else:
                    current_entity = entity_label[2:]
                    current_text += " " + word

        # add the last entity if exists
        if current_entity is not None:
            processed_results.append({'entity': current_entity, 'text': current_text.strip()})

        # map specific entities to desired labels
        label_mapping = {
            'PERSON': 'PER',
            'ORGANIZATION': 'ORG',
            'LOCATION': 'LOC'
            # add more entity types as needed
        }

        for result in processed_results:
            entity_label = result['entity']
            if entity_label in label_mapping:
                result['entity'] = label_mapping[entity_label]

        return processed_results

    def replace_entities(self, original_text, processed_results):
        def remove_special_tokens(text):
            cleaned_text = re.sub(r'(?:(?<!\s)##)|(\s##)', '', text)
            return cleaned_text

        for result in processed_results:
            result['text'] = remove_special_tokens(result['text'])

        def replace_text_with_entity(original_text, entity_result):
            entity_label = entity_result['entity']
            entity_text = entity_result['text']
            return original_text.replace(entity_text, entity_label)

        processed_text = original_text
        for result in processed_results:
            processed_text = replace_text_with_entity(processed_text, result)

        return processed_text

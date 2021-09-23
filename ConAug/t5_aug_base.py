import re
import ConAug.noising as noising


def create_sentinel_dict(output, tokenizer, pattern="<extra_id_\d+>"):
    sentinel_dict = dict()
    tokenized_output = tokenizer.tokenize(output)
    all_sentinel_tokens = re.findall(pattern, output)
    key = ""; value = ""
    for token in tokenized_output:
        if token in all_sentinel_tokens:
            if key != "":
                sentinel_dict[key] = value
            key = token; value = ""
        else:
            value += token
    return sentinel_dict


def format_output(input, sentinel_dict):
    # Find actual sentinel tokens in input
    input_sentinel_tokens = re.findall("<extra_id_\d*>", input)
    # Replace them with output
    for token in input_sentinel_tokens:
        if token in sentinel_dict:
            input = input.replace(token, sentinel_dict[token])
        else:
            input = input.replace(token, "")
    return input.replace('▁', ' ').replace('</s>', '')


def context_aug_batch(sentence_batch, tokenizer, mlm, device):
    masked_batch = list()
    augmented_batch = list()
    for sentence in sentence_batch:
        tokenized_sentence = tokenizer.encode(sentence, truncation=True, max_length=512)  # check this
        masked_sentence, _ = noising.add_noise(tokenized_sentence, tokenizer)
        masked_batch.append(masked_sentence)
    encoded_masked_batch = tokenizer(masked_batch, padding=True, add_special_tokens=True, return_tensors='pt')
    encoded_masked_batch = encoded_masked_batch['input_ids'].to(device)
    output = mlm.generate(input_ids=encoded_masked_batch, num_beams=2, num_return_sequences=1)
    for i in range(len(output)):
        decoded = tokenizer.decode(output[i])
        sd = create_sentinel_dict(decoded, tokenizer)
        aug = format_output(masked_batch[i], sd).rstrip()
        augmented_batch.append(aug)
    return augmented_batch

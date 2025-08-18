import os
import json


def main():

    if not os.path.isfile('./data/bn_to_lemmas.json'):

        bn_to_lemmas = {}

        with open("./data/lemmas_OK.jsonl", "r", encoding="utf8") as f:
            for l in f:
                line_data = json.loads(l)

                if len(line_data["lemmas"]) != 6:
                    continue
        
                bn_to_lemmas[line_data["id"]] = line_data["lemmas"]
        
        with open('./data/bn_to_lemmas.json', "w", encoding="utf8") as f:
            json.dump(bn_to_lemmas, f, indent=2)
        
        for lang in ["en", "it", "fr", "de", "es", "fa"]:

            lemma_to_bns = {}

            for x, y in bn_to_lemmas.items():

                for lemma in y[lang]:

                    lemma = lemma.replace("_", " ")

                    if lemma not in lemma_to_bns:
                        lemma_to_bns[lemma] = []
                    
                    lemma_to_bns[lemma].append(x)
            
            with open(f'./data/lemma_to_bn/{lang}.json', 'w', encoding="utf8") as f:
                json.dump(lemma_to_bns, f, indent=2)
    

if __name__ == "__main__":

    main()
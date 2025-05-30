import ast
import re
import numpy as np
from collections import Counter
from evaluate import load

def ensure_list(obj):
    """
    Se obj è una lista, la restituisce.
    Se obj è una stringa, prova a fare ast.literal_eval.
    In caso di errore o tipo diverso, restituisce [].
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        try:
            return ast.literal_eval(obj)
        except (ValueError, SyntaxError):
            return []
    return []

# ===== FUNZIONI prompt23_14 ed eval23_14 =====

def prompt23_14(row):
    domanda = row["question"]
    scelte = row["options"]
    contesto = row["context"]
    prompt = ""
    if contesto is not np.nan:
        prompt += f"Contesto: {contesto}\n\n"
    prompt += f"Domanda:\n{domanda}\n\n"
    if scelte:
        prompt += f"Opzioni:\n{scelte}\n\n"
    prompt += "Istruzioni: \n"
    prompt += (
        "Scrivi le parole scelte nell'ordine in cui devono essere inserite "
        "negli spazi vuoti. Formatta la tua risposta come mostrato nell'esempio qui sotto.\n"
        "Formato della risposta: [parola1, parola2, parola3] \n\n"
        "Risposta: "
    )
    return prompt

def eval23_14(model_output, true_answer):
    contenuto = re.findall(r"\[(.*?)\]", model_output)
    if not contenuto:
        return 0

    contenuto_pulito = contenuto[0].replace("'", "").replace('"', "").split(", ")
    word_list = [word.lower() for word in contenuto_pulito]

    # Convertiamo true_answer in una lista
    target = ensure_list(true_answer)
    # Ora 'target' dovrebbe essere, ad esempio, ["parola1", "parola2", "parola3"]
    target = list(map(str.lower, target))

    if target == word_list:
        return 1
    else:
        return 0


# ===== FUNZIONI prompt24_14 ed eval24_14 =====

def prompt24_14(row):
    domanda = row["question"]
    contesto = row["context"]
    prompt = ""
    if contesto is not np.nan:
        prompt += f"Contesto: {contesto}\n\n"
    prompt += f"Domanda:\n{domanda}\n\n "
    scelte = row["options"]
    if scelte:
        prompt += f"Opzioni:\n{scelte}\n\n"
    prompt += "Istruzioni: \n"
    prompt += (
        "Riscrivi la frase nel modo corretto e non aggiungere nient'altro.\n\n"
        "Risposta: "
    )
    return prompt

def eval24_14(model_output, true_answer):
    # true_answer potrebbe essere una lista di possibili pattern
    patterns_list = ensure_list(true_answer)
    # Se non otteniamo liste di stringhe, fallback a 0
    if not patterns_list:
        return 0

    # Escape each pattern to safely handle special characters
    escaped_patterns = [re.escape(pattern) for pattern in patterns_list]
    combined_pattern = "|".join(escaped_patterns)

    # Search for the pattern in the input text
    if re.search(combined_pattern, model_output):
        return 1
    else:
        return 0


# ===== FUNZIONI prompt26_14 ed eval26_14 =====

def prompt26_14(row):
    domanda = row["question"]
    contesto = row["context"]
    prompt = ""
    if contesto is not np.nan:
        prompt += f"Contesto: {contesto}\n\n"
    prompt += f"Domanda:\n{domanda}\n\n "
    scelte = row["options"]
    if scelte:
        prompt += f"Opzioni:\n{scelte}\n\n"
    prompt += (
        "Istruzioni: \n"
        "Devi restituire la parola corrispondente alla risposta esatta tra parentesi quadre.\n"
        "Formato della risposta: [parola] \n\n"
        "Risposta: "
    )
    return prompt

def eval26_14(model_output, true_answer):
    contenuto = re.findall(r"\[(.*?)\]", model_output)
    if not contenuto:
        return 0

    contenuto_text = " ".join(contenuto).lower()

    # Ensure that true_answer is a list
    answer_list = ensure_list(true_answer)
    # Potrebbe essere un array di 1 elemento (es: ["pippo"]), o più
    for ans in answer_list:
        ans_lower = ans.lower()
        escaped_answer = re.escape(ans_lower)
        if re.search(escaped_answer, contenuto_text):
            return 1
    return 0


# ===== FUNZIONI prompt29_14 ed eval29_14 =====

def prompt29_14(row):
    domanda = row["question"]
    scelte = row["options"]
    contesto = row["context"]
    prompt = ""
    if contesto is not np.nan:
        prompt += f"Contesto: {contesto}\n\n"
    prompt += f"Domanda:\n{domanda}\n\n "
    if scelte:
        prompt += f"Opzioni:\n{scelte}\n\n"
    prompt += (
        "Istruzioni: \n"
        "Scrivi le parole scelte nelle parentesi quadre. Formatta la tua risposta come mostrato nell'esempio qui sotto.\n"
        "Formato della risposta: [parola1, parola2, parola3] \n\n"
        "Risposta: "
    )
    return prompt

def eval29_14(model_output, true_answer):
    contenuto = re.findall(r"\[(.*?)\]", model_output)
    if not contenuto:
        return 0

    contenuto_pulito = contenuto[0].replace("'", "").replace('"', "").split(", ")
    word_list = [word.lower() for word in contenuto_pulito]

    # Converti true_answer in una lista
    target_list = ensure_list(true_answer)
    # es: ["parola1", "parola2", "parola3"] (oppure una sola, ecc.)
    target_list = [t.lower() for t in target_list]

    # Utilizza Counter per confrontare gli elementi senza considerare l'ordine
    if Counter(target_list) == Counter(word_list):
        return 1
    else:
        return 0


# ===== prompt21_15 => prompt24_14, eval21_15 => eval24_14 =====
def prompt21_15(row):
    return prompt24_14(row)

def eval21_15(model_output, true_answer):
    return eval24_14(model_output, true_answer)


# ===== prompt35_15 ed eval35_15 =====

def prompt35_15(row):
    contesto = row["context"]
    domanda = row["question"]
    prompt = ""
    if contesto is not np.nan:
        prompt += f"Contesto: {contesto}\n\n"
    prompt += f"Domanda:\n{domanda}\n\n "
    scelte = row["options"]
    if scelte:
        prompt += f"Opzioni:\n{scelte}\n\n"
    prompt += "Risposta: "
    return prompt

def eval35_15(model_output, true_answer, threshold=0.7):
    # true_answer potrebbe essere una lista di frasi da confrontare con BERTScore
    answer_list = ensure_list(true_answer)

    bertscore = load("bertscore")
    predictions = [model_output] * len(answer_list)

    # se answer_list è una lista di stringhe
    results = bertscore.compute(predictions=predictions, references=answer_list, lang="it")
    max_f1 = max(results["f1"])

    if max_f1 >= threshold:
        return 1
    else:
        return 0


# ===== prompt54_15 ed eval54_15 =====

def prompt54_15(row):
    contesto = row["context"]
    domanda = row["question"]
    prompt = ""
    if contesto is not np.nan:
        prompt += f"Contesto: {contesto}\n\n"
    prompt += f"Domanda:\n{domanda}\n\n "
    scelte = row["options"]
    if scelte:
        prompt += f"Opzioni:\n{scelte}\n\n"
    prompt += (
        "Istruzioni: \n"
        "Scrivi le parole scelte nelle parentesi quadre. Formatta la tua risposta come mostrato nell'esempio qui sotto.\n"
        "Formato della risposta: [parola1, parola2, parola3] \n\n"
        "Risposta: "
    )
    return prompt

def eval54_15(model_output, true_answer):
    contenuto = re.findall(r"\[(.*?)\]", model_output)
    if not contenuto:
        return 0

    contenuto_pulito = contenuto[0].replace("'", "").replace('"', "").split(", ")
    word_list = [w.lower() for w in contenuto_pulito]

    # Converte la stringa di risposta vera in una lista di liste
    # es: [[...],[...],...]
    target_list = ensure_list(true_answer)

    for answer in target_list:
        # answer dovrebbe essere una lista, es. ["parola1","parola2","parola3"]
        if not isinstance(answer, list):
            # skip se non è una lista
            continue
        answer_lower = [a.lower() for a in answer]
        if Counter(answer_lower) == Counter(word_list):
            return 1
    return 0


# ===== prompt55_15 ed eval55_15 =====

def prompt55_15(row):
    contesto = row["context"]
    domanda = row["question"]
    prompt = ""
    if contesto is not np.nan:
        prompt += f"Contesto: {contesto}\n\n"
    prompt += f"Domanda:\n{domanda}\n\n "
    scelte = row["options"]
    if scelte:
        prompt += f"Opzioni:\n{scelte}\n\n"
    prompt += (
        "Istruzioni: \n"
        "Devi restituire la parola corrispondente alla risposta esatta tra parentesi quadre.\n"
        "Formato della risposta: [parola] \n\n"
        "Risposta: "
    )
    return prompt

def eval55_15(model_output, true_answer):
    contenuto = re.findall(r"\[(.*?)\]", model_output)

    # Prepara la lista di risposte corrette
    answer_list = ensure_list(true_answer)
    answer_list = [a.lower() for a in answer_list]

    if not contenuto:
        return 0

    # Pulisce
    contenuto_pulito = contenuto[0].replace("'", "").replace('"', "")
    word = contenuto_pulito.lower()

    # Controllo se word e' in answer_list
    if word in answer_list:
        return 1
    else:
        return 0


def prompt57_15(row):
    return prompt35_15(row)

def eval57_15(model_output, true_answer):
    return eval35_15(model_output, true_answer)


# ===== prompt10_17 => prompt55_15, eval10_17 => eval55_15

def prompt10_17(row):
    domanda = row["question"]
    contesto = row["context"]
    prompt = ""
    if contesto is not np.nan:
        prompt += f"Contesto: {contesto}\n\n"
    prompt += f"Domanda:\n{domanda}\n\n "
    scelte = row["options"]
    if scelte:
        prompt += f"Opzioni:\n{scelte}\n\n"
    prompt += (
        "Istruzioni: \n"
        "Devi restituire il completamento della frase tra parentesi quadre.\n"
        "Formato della risposta: [completamento] \n\n"
        "Risposta: "
    )
    return prompt

def eval10_17(model_output, true_answer):
    return eval55_15(model_output, true_answer)

# analogamente per prompt19_17 => prompt55_15, ecc.

def prompt19_17(row):
    return prompt55_15(row)

def eval19_17(model_output, true_answer):
    return eval55_15(model_output, true_answer)

def prompt1_30(row):
    return prompt26_14(row)

def eval1_30(model_output, true_answer):
    return eval26_14(model_output, true_answer)

def prompt2_30(row):
    return prompt26_14(row)

def eval2_30(model_output, true_answer):
    return eval26_14(model_output, true_answer)

def prompt3_30(row):
    return prompt26_14(row)

def eval3_30(model_output, true_answer):
    return eval26_14(model_output, true_answer)

def prompt4_30(row):
    return prompt26_14(row)

def eval4_30(model_output, true_answer):
    return eval26_14(model_output, true_answer)

def prompt34_30(row):
    return prompt23_14(row)

def eval34_30(model_output, true_answer):
    return eval23_14(model_output, true_answer)

def prompt51_30(row):
    return prompt35_15(row)

def eval51_30(model_output, true_answer):
    return eval35_15(model_output, true_answer, 0.68)

def prompt52_30(row):
    return prompt35_15(row)

def eval52_30(model_output, true_answer):
    return eval35_15(model_output, true_answer, 0.68)

def prompt53_30(row):
    return prompt55_15(row)

def eval53_30(model_output, true_answer):
    return eval24_14(model_output, true_answer)

def prompt55_30(row):
    return prompt55_15(row)

def eval55_30(model_output, true_answer):
    return eval24_14(model_output, true_answer) #cambia: matcha la parola bisanzio

def prompt58_30(row):
    return prompt35_15(row)

def eval58_30(model_output, true_answer):
    return eval35_15(model_output, true_answer)

def prompt69_30(row):
    return prompt55_15(row)

def eval69_30(model_output, true_answer):
    return eval24_14(model_output, true_answer)

def prompt71_30(row):
    return prompt35_15(row)

def eval71_30(model_output, true_answer):
    return eval35_15(model_output, true_answer)

def prompt73_30(row):
    contesto = row["context"]
    domanda = row["question"]
    prompt = ""
    if contesto is not np.nan:
        prompt += f"Contesto: {contesto}\n\n"
    prompt += f"Domanda:\n{domanda}\n\n "
    scelte = row["options"]
    if scelte:
        prompt += f"Opzioni:\n{scelte}\n\n"
    prompt += (
        "Istruzioni: \n"
        "Devi restituire la frase corrispondente alla risposta esatta tra parentesi quadre.\n"
        "Formato della risposta: [frase] \n\n"
        "Risposta: "
    )
    return prompt

def eval73_30(model_output, true_answer):
    return eval24_14(model_output, true_answer)

def prompt74_30(row):
    return prompt35_15(row)

def eval74_30(model_output, true_answer):
    return eval35_15(model_output, true_answer)

def prompt75_30(row):
    return prompt54_15(row)

def eval75_30(model_output, true_answer):
    contenuto = re.findall(r"\[(.*?)\]", model_output)
    if not contenuto:
        return 0

    contenuto_pulito = contenuto[0].replace("'", "").replace('"', "").split(", ")
    word_list = [w.lower() for w in contenuto_pulito]

    if len(word_list) != 3:
        return 0

    if len(set(word_list)) != len(word_list):
        return 0

    # target potrebbe essere una lista di stringhe
    target_list = ensure_list(true_answer)
    target_list = [str(t).lower() for t in target_list]

    # Verifica se *tutti* gli item in word_list sono in target_list
    # (come da tuo codice: all([item in target for item in word_list]))
    if all(item in target_list for item in word_list):
        return 1
    else:
        return 0

def prompt39_49(row):
    return prompt26_14(row)

def eval39_49(model_output, true_answer):
    return eval19_17(model_output, true_answer)

def prompt40_49(row):
    return prompt26_14(row)

def eval40_49(model_output, true_answer):
    return eval19_17(model_output, true_answer)

def prompt41_49(row):
    return prompt26_14(row)

def eval41_49(model_output, true_answer):
    return eval19_17(model_output, true_answer)

def prompt45_49(row):
    return prompt26_14(row)

def eval45_49(model_output, true_answer):
    return eval19_17(model_output, true_answer)

def prompt51_53(row):
    return prompt55_15(row)

def eval51_53(model_output, true_answer):
    return eval19_17(model_output, true_answer)

def prompt52_53(row):
    return prompt55_15(row)

def eval52_53(model_output, true_answer):
    return eval55_15(model_output, true_answer)

def prompt53_53(row):
    return prompt55_15(row)

def eval53_53(model_output, true_answer):
    return eval19_17(model_output, true_answer)

def prompt54_53(row):
    return prompt55_15(row)

def eval54_53(model_output, true_answer):
    return eval19_17(model_output, true_answer)

def prompt25_57(row):
    return prompt35_15(row)

def eval25_57(model_output, true_answer):
    return eval24_14(model_output, true_answer)

def prompt27_57(row):
    return prompt55_15(row)

def eval27_57(model_output, true_answer):
    return eval19_17(model_output, true_answer)

def prompt28_57(row):
    return prompt55_15(row)

def eval28_57(model_output, true_answer):
    return eval55_15(model_output, true_answer)

def prompt48_57(row):
    return prompt55_15(row)

def eval48_57(model_output, true_answer):
    return eval19_17(model_output, true_answer)



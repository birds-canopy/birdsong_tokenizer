from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
import pandas as pd
import numpy as np
import tempfile
import os
import sentencepiece as spm 
from src.utils import remove_consecutive_duplicate_words, zipf_law_fit, shannon_entropy, flatten_chunks_as_sequences, corpus_to_string, tuple_to_string
from collections import Counter
# sum = __builtins__.sum  # Utiliser le sum de Python pour éviter les conflits avec d'autres fonctions (pb que j'ai eu à un moment, c'est peut-être plus utile)

def tokenize_syllables(Y, vocab_size, tokenizer_type):
    """Tokenize syllables Y into a list of lists of tokens.

    Args:
        Y (pd.Series): Série de syllabes.
        vocab_size (int): Taille du vocabulaire.
        tokenizer_type (str): Type de tokenizer à utiliser ("BPE", "WordPiece", "Unigram", "SentencePiece").

    Returns:
        pd.DataFrame: DataFrame contenant les tokens sous forme brute.
    """


    Y_c = Y.copy()
    Y_c.replace('St', np.nan, inplace=True)

    data_tokenizer = []
    chant_cache = ""

    for i in range(len(Y_c)):
        syllabe = str(Y[i])
        if syllabe not in ['SIL', 'call', 'cri', 'TRASH', 'None', 'nan']:
            if i > 0:
                prev_syllabe = str(Y_c[i - 1])
                if syllabe == 'E2' and prev_syllabe == 'E1':
                    Y_c[i - 1] = ''
                    syllabe = 'E'
                elif syllabe in ['E1', 'E2']:
                    syllabe = 'E'
                if syllabe == 'S2' and prev_syllabe == 'S1':
                    syllabe = 'S'
                    Y_c[i - 1] = ''
                elif syllabe in ['S1', 'S2']:
                    syllabe = 'S'
                if syllabe == 'P2' and prev_syllabe == 'P1':
                    syllabe = 'P'
                    Y_c[i - 1] = ''
                elif syllabe in ['P1', 'P2']:
                    syllabe = 'P'
                if syllabe == 'A2' and prev_syllabe == 'A1':
                    syllabe = 'A'
                    Y_c[i - 1] = ''
                elif syllabe in ['A1', 'A2']:
                    syllabe = 'A'
                if syllabe in ['St', 'St2']:
                    syllabe = 'S'
                if syllabe == 'Ti':
                    syllabe = 'T'

            if syllabe != 'END':
                chant_cache += syllabe + " "
            else:
                if chant_cache:
                    data_tokenizer.append(chant_cache.strip())
                chant_cache = ""

    if chant_cache:
        data_tokenizer.append(chant_cache.strip())

    data_tokenizer = [remove_consecutive_duplicate_words(seq) for seq in data_tokenizer]
    # with open("data_rouge6_vclean.txt", "w", encoding="utf-8") as f:
    #     for line in data_tokenizer:
    #         line = "START " + line + " END"
    #         f.write(line + "\n")
    raw_tokenized = []
    id_tokenized = []

    if tokenizer_type in ["BPE", "WordPiece", "Unigram"]:
        if tokenizer_type == "BPE":
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]"], vocab_size=vocab_size)
        elif tokenizer_type == "WordPiece":
            tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
            trainer = WordPieceTrainer(special_tokens=["[UNK]", "[PAD]"], vocab_size=vocab_size)
        elif tokenizer_type == "Unigram":
            tokenizer = Tokenizer(Unigram())
            trainer = UnigramTrainer(special_tokens=["[UNK]", "[PAD]"], vocab_size=vocab_size)

        tokenizer.train_from_iterator(data_tokenizer, trainer)
        tokenizer.save(f"tokenizer-{tokenizer_type}-train.json")

        for sequence in data_tokenizer:
            token = tokenizer.encode(sequence)
            raw_tokenized.append(token.tokens)
            id_tokenized.append(token.ids)

    elif tokenizer_type == "SentencePiece":
        # Entraînement SentencePiece
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as temp_file:
            for line in data_tokenizer:
                temp_file.write(line + "\n")
            temp_path = temp_file.name

        model_prefix = "spm_tokenizer"
        spm.SentencePieceTrainer.Train(
            f"--input={temp_path} --model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} --model_type=unigram --pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1 --hard_vocab_limit=false"
        )

        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")

        for sequence in data_tokenizer:
            raw_tokenized.append(sp.encode(sequence, out_type=str))
            id_tokenized.append(sp.encode(sequence, out_type=int))

        os.remove(temp_path)

    else:
        raise ValueError(f"Tokenizer type '{tokenizer_type}' not supported")

    syll_token = pd.DataFrame(raw_tokenized).astype(str).replace(to_replace=r'SIL', value='', regex=True)
    id_token = pd.DataFrame(id_tokenized)
    id_token = id_token.where(syll_token.notna(), 0)

    return syll_token

def analyze_vocab_sizes(tokenize_syllables, syllable_data, vocab_sizes, tokenizer_type="BPE"):
    """"Analyse des tailles de vocabulaire et des occurrences de tokens.
    Args:
        tokenize_syllables (function): Fonction de tokenisation.
        syllable_data (pd.Series): Série de syllabes à tokeniser.
        vocab_sizes (range): Plage de tailles de vocabulaire à analyser.
        tokenizer_type (str): Type de tokenizer à utiliser ("BPE", "WordPiece", "Unigram", "SentencePiece"). 
    Returns:
        pd.DataFrame: DataFrame contenant les statistiques des chunks pour chaque taille de vocabulaire.
        new_token_occurrence_df (pd.DataFrame): DataFrame contenant les nouveaux tokens et leurs occurrences pour chaque taille de vocabulaire. 
    """  
    chunks = None
    chunks_vocab_33 = None # Debug
    if tokenizer_type not in ["BPE", "WordPiece", "Unigram", "SentencePiece"]:
        raise ValueError(f"Tokenizer type '{tokenizer_type}' not supported")
    results = []
    new_token_occurrence_all = []  # Liste pour stocker les nouveaux tokens et leurs occurrences avec vocab_size

    for vocab_size in vocab_sizes:
        if vocab_size > min(vocab_sizes):
            prev_chunks = chunks.copy(deep=True)

        tokenized_df = tokenize_syllables(syllable_data, vocab_size, tokenizer_type)

        #################### ZIPF LAW ####################
        flat_tokens = []


        for cell in tokenized_df.values.flatten():
            if isinstance(cell, str) and cell != 'None':
                token = cell.replace('#', '').strip()
                token = ' '.join(token.split())  # supprime les doubles espaces internes et débuts/fin
                if 2 <= len(token.split()) <= 6:
                    flat_tokens.append(token)
        if len(flat_tokens) > 0:
            slope, intercept, r2 = zipf_law_fit(flat_tokens)
        else : 
            slope, intercept, r2 = 0, 0, 0


        tokenized_df = tokenized_df.dropna()
        tokenized_df = tokenized_df.replace('None', ' ', regex=True)
        # print(f"--- vocab_size={vocab_size} ---")
        # print(tokenized_df.head(1))

        # Conversion en liste
        # chunks = tokenized_df.applymap(lambda x: str(x).split() if pd.notna(x) else [])
        chunks = tokenized_df.apply(lambda col: col.map(lambda x: str(x).split() if pd.notna(x) else []))
        if tokenizer_type == "WordPiece":
            def clean_token_list(token_list):
                """Nettoie une liste de tokens WordPiece en supprimant les préfixes ## et #"""
                if isinstance(token_list, list):
                    cleaned = []
                    for token in token_list:
                        if isinstance(token, str):
                            # Supprimer tous les "#" (y compris "##")
                            clean_token = token.replace('##', '').replace('#', '').strip()
                            # Supprimer les espaces multiples
                            clean_token = ' '.join(clean_token.split())
                            if clean_token:  # Ajouter seulement si le token n'est pas vide
                                cleaned.append(clean_token)
                    return cleaned
                return token_list
            
            chunks = chunks.applymap(clean_token_list)





        # chunks = tokenized_df.applymap(lambda x: x if isinstance(x, list) else [])

        # print(f"Shape of tokenized_df.values[0][0]: {len(tokenized_df.values[0][0]) if len(tokenized_df.values) > 0 
        # and len(tokenized_df.values[0]) > 0 else 'N/A'}")
        token_counts = Counter(flat_tokens)
        assert isinstance(token_counts, dict)

        ###################### ENTROPIE DE SHANNON ######################
        # Convertir les comptes de tokens en distribution de probas
        total_tokens = sum(token_counts.values())
        # total_tokens = __builtins__.sum(token_counts.values()) # Mon sum bug ?????

        # print("type(sum):", type(sum))
        print("token_counts.values():", token_counts.values())
        # print("total_tokens:", total_tokens)
        token_probs = [count / total_tokens for count in token_counts.values() if count > 0]
        print("token_probs:", token_probs)
        print("mean token probs :", np.mean(token_probs))
        # Calculer l'entropie de Shannon

        entropy_val = shannon_entropy(token_probs) 



        # print(f"[DEBUG] Token counts for vocab_size {vocab_size}: {token_counts}")
        if vocab_size > min(vocab_sizes):

            prev_tokens = flatten_chunks_as_sequences(prev_chunks)
            current_tokens = flatten_chunks_as_sequences(chunks)
            new_token = current_tokens - prev_tokens
            if new_token:
                # Convertir le corpus en une longue chaîne de caractères
                corpus_string = corpus_to_string(chunks)

                # Convertir le new_token en string : {('Z', 'A')} -> 'ZA'
                new_token_str = [tuple_to_string(token) for token in new_token]

                # Compter les occurrences de chaque nouveau token dans la chaîne du corpus
                new_token_occurrence = {token: corpus_string.count(token) for token in new_token_str}

                # Stocker les résultats dans une liste sous forme de tuples
                for token, occurrence in new_token_occurrence.items():
                    new_token_occurrence_all.append({
                        'vocab_size': vocab_size,
                        'token': token,
                        'occurrence': occurrence
                    })
                    # if vocab_size == 60:
                    #     print(f"[DEBUG] occurence of {token}: {occurrence}")

                # Debugging output pour vérifier que les occurrences sont correctement comptées
                # print(f"[DEBUG] new_token_occurrence for vocab_size {vocab_size}: {new_token_occurrence}")

        # Calcul des longueurs des chunks
        # chunk_lengths = chunks.applymap(lambda x: [len(chunk) for chunk in x] if isinstance(x, list) else [])
        chunk_lengths = chunks.apply(lambda col: col.map(lambda x: len(x) if isinstance(x, list) else 0))

        # all_lengths = [length for sublist in chunk_lengths.values.flatten()
        #                if isinstance(sublist, list) for length in sublist]
        all_lengths = [length for length in chunk_lengths.values.flatten() if length > 0]


        if not all_lengths:
            continue

        # Statistiques
        min_length = np.min(all_lengths)
        max_length = np.max(all_lengths)
        mean_length = np.mean(all_lengths)
        # q25_length = np.percentile(all_lengths, 25)
        # q75_length = np.percentile(all_lengths, 75)

        print(f"vocab_size={vocab_size} | mean_length={mean_length:.2f} | nb_chunks={len(all_lengths)}")
        length_counts = pd.Series(all_lengths).value_counts()

        min_occurrences = length_counts.get(min_length, 0)
        max_occurrences = length_counts.get(max_length, 0)
        mean_occurrences = length_counts.get(int(mean_length), 0)
        # q25_occurrences = length_counts.get(int(q25_length), 0)
        # q75_occurrences = length_counts.get(int(q75_length), 0)
                # Quantiles détaillés
        quantiles = list(range(10, 101, 10)) + [25, 75]  # q10 to q100 + q25, q75
        quantile_lengths = {f"q{q}_length": np.percentile(all_lengths, q) for q in quantiles}
        quantile_occurrences = {
            f"q{q}_occurrences": length_counts.get(int(np.percentile(all_lengths, q)), 0) for q in quantiles
        }

        if vocab_size == 33:
            chunks_vocab_33 = chunks.copy(deep=True)
        results.append({
            "vocab_size": vocab_size,
            "min_length": min_length,
            "min_occurrences": min_occurrences,
            "max_length": max_length,
            "max_occurrences": max_occurrences,
            "mean_length": mean_length,
            "mean_occurrences": mean_occurrences,
            # "q25_length": q25_length,
            # "q25_occurrences": q25_occurrences,
            # "q75_length": q75_length,
            # "q75_occurrences": q75_occurrences, 
            "r2": r2, 
            "slope": slope,
            "intercept": intercept, 
            "entropy": entropy_val, 
            **quantile_lengths,
            **quantile_occurrences
        })


    new_token_occurrence_df = pd.DataFrame(new_token_occurrence_all)

    return pd.DataFrame(results), new_token_occurrence_df, chunks, chunks_vocab_33
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.stats import linregress
import math
import sys
import os
import itertools


sys.path.append(os.path.abspath("../src"))


# Pre process

#Transformation d'un dataframe contenant les annotations des sylalbes du chant en une séquence de labels temporels
def df_to_seq(df, length_enregis=23079):
    
    """ From a dataframe containing, for all phrases of a birdsong, at least three columns:
    the start of the song in seconds, the end of the song in seconds, the type of the syllable/phrase. 
    Convert it to a sequence like [A A A]."""
    
    df_seq=df.copy(deep=True)
    df_seq['start']=df['start']*1000 #convert to ms
    df_seq['end']=df['end']*1000

    seq=[]
    end_prec=0
    
    for line in df_seq.index:
        # add 'SIL', silence label, 
        # if the timestep of the end of the previous phrase isn't the same as  the 
        # timestep of the start of the phrase, meaning there is silence between the two phrases
        if end_prec != df_seq['start'][line]: 
            nb_sil=df_seq['start'][line]-end_prec
            seq.extend('SIL'for i in range(int(nb_sil)))

        # otherwise just add the syllable in the sequence, times the number of ms
        nb_syll=df_seq['end'][line]-df_seq['start'][line]
        seq.extend(df_seq['syll'][line] for i in range(int(nb_syll)))
        end_prec=df_seq['end'][line]
    
    seq.extend('SIL' for i in range(length_enregis-len(seq))) # adding the silence between the end of the last phrase and the end of the recording
   
    return seq

def aggregating_labels(df,process='all_ms',nb_ms=1, length_enregis=23079):
    """ Returns y, all the ground truth labels for one file.
    Three different processes are implemented:
    * 'all_ms': the label of the syllables for every ms
    * 'phrase': labels of each phrase of the song, containing silences, ordered but indifferent to each length
    * 'ms_pack': labels most present in group of a number of n ms, the argument 'nb_ms' defines the number of ms
    * 'phrase_start': labels of each phrase of the song, containing silences, ordered but indifferent to each length, same as 'phrase'
    """
    
    seq = df_to_seq(df, length_enregis=length_enregis) 
    seq.extend('SIL' for i in range(length_enregis-len(seq)))
    
    if process=='ms_pack': # nb_ms is the lenghth of the window, in which we proceed as a majority vote
        
        ys=[]

        for i in range(len(seq)//nb_ms):

            start_win=i*nb_ms
            win = seq[start_win:start_win+nb_ms]
            sylls = [(k, len(list(g))) for k, g in itertools.groupby(win)] 
            sylls_dict=dict(sylls)
#             print(sylls_dict)
            
            best_syll = max(sylls_dict, key = lambda k: sylls_dict[k]) # the most present syllable type in the window is attributed for this group of ms
#             print(best_syll)

            ys.append(best_syll)
        
        y=ys
        
    elif process== 'phrase' or process=='phrase_start':
        df_seq = df.copy(deep=True)
        starts = df_seq['start'].to_numpy()
        ends = df_seq['end'].to_numpy()
        sylls = df_seq['syll'].to_numpy()
        
        ys=[]
        end_prec=0
        for i,start in enumerate(starts):

            if start != end_prec:
                ys.append('SIL') #for silences

            ys.append(sylls[i])
            end_prec=ends[i]

        ys.append('SIL')

        y=ys

        
    elif process != 'all_ms':
        y=("wrong process entered, try another one")
        
    else:
        y = seq # all_ms is the default value of the argument process, it corresponds to the output of the function df_to_seq
        
    return y 
def process_Y_data( songs_path, process, nb_ms, aggregated_trials, nb_trials):
    Y = []  # ground truth, label for each ms
    for song in os.listdir(songs_path):  # we iterate through per song
        song_wo_ext = song[:-4]
        song_path = os.path.join(songs_path, song_wo_ext + ".csv")
        df = pd.read_csv(song_path)

        y= aggregating_labels(df, process=process, nb_ms=nb_ms)
        y.append('END')
        if not aggregated_trials:
            for i in range(nb_trials):
                Y.extend(y)
        else:
            Y.extend(y)  # for each file, it adds the labels (aggregated or not) to the general Y vector
        
    return Y

def clean_Y(Y):
    """
    Method to clean rouge6 labels
    """
    Y_c = Y.copy()  
    data_clean = []


    for i in range(len(Y)):
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
                if syllabe == 'END':
                    syllabe = "%"
                if syllabe == 'START':
                    syllabe = "#"
            if syllabe == '%':
                data_clean.append(syllabe)
                data_clean.append("#")
            else :
                data_clean.append(syllabe)
        elif i == 0:
            data_clean.append("#")

    # Remove consecutive duplicates
    for i in range(len(data_clean) - 1, 0, -1):
        if data_clean[i] == data_clean[i - 1]:
            data_clean.pop(i)  

    return data_clean



# Utils
def remove_consecutive_duplicate_words(s):
    words = s.split()
    cleaned = [words[0]] if words else []
    for word in words[1:]:
        if word != cleaned[-1]:
            cleaned.append(word)
    return ' '.join(cleaned)

######################### utils plot conv div ##################################################
def find_optimal_path_label_position(start_coord, end_coord, existing_positions,
                                    min_distance=0.06, side=None):
    mid_x = (start_coord[0] + end_coord[0]) / 2
    mid_y = (start_coord[1] + end_coord[1]) / 2
    dx = end_coord[0] - start_coord[0]
    dy = end_coord[1] - start_coord[1]
    length = np.hypot(dx, dy)

    if length > 0:
        perp_x = -dy / length
        perp_y = dx / length

        if side is not None:
            candidate_x = mid_x + side * 0.05 * perp_x
            candidate_y = mid_y + side * 0.05 * perp_y
            return candidate_x, candidate_y

        offsets = [0.03, -0.03, 0.05, -0.05, 0.07, -0.07]
        for offset in offsets:
            candidate_x = mid_x + offset * perp_x
            candidate_y = mid_y + offset * perp_y
            if all(np.hypot(candidate_x - ex_x, candidate_y - ex_y) >= min_distance
                for ex_x, ex_y in existing_positions):
                return candidate_x, candidate_y

    return mid_x, mid_y

def find_optimal_label_position(x, y, existing_positions, min_distance=0.08):
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)  
    distances = np.linspace(0.02, 0.15, 7)  

    for dist in distances:
        for angle in angles:
            candidate_x = x + dist * np.cos(angle)
            candidate_y = y + dist * np.sin(angle)
            if all(np.hypot(candidate_x - ex_x, candidate_y - ex_y) >= min_distance
                for ex_x, ex_y in existing_positions):
                return candidate_x, candidate_y

    return x - 0.05, y - 0.05

def calculate_transition_probabilities(sequence, order=1, seuil=50):
    """
    Calcule les probabilités de transition entre chaque paire de caractères.

    p_div = P(char_b | char_a)
    p_conv = P(char_a | char_b)
    """

    transitions = defaultdict(int)
    char_counts = Counter(sequence)
    is_between_2_songs = False

    for i in range(len(sequence) - order):
        src = sequence[i]
        tgt = sequence[i + order]
        for j in range(order):
            if i + j + 1 < len(sequence):
                next_tgt = sequence[i + j + 1]
                if next_tgt == '%':
                    is_between_2_songs = True
                elif next_tgt == '#':
                    is_between_2_songs = True
        if not is_between_2_songs:
            transitions[(src, tgt)] += 1
        else : 
            is_between_2_songs = False
    
    unique_chars = list(set(sequence))
    results = []

    for char_a in unique_chars:
        for char_b in unique_chars:
            # if char_a != char_b:
            count_ab = transitions.get((char_a, char_b), 0)
            if count_ab >= seuil:
                total_a = char_counts[char_a]
                total_b = char_counts[char_b]
                
                p_div = count_ab / total_a if total_a > 0 else 0
                p_conv = count_ab / total_b if total_b > 0 else 0

                results.append({
                    'couple': f"{char_a}-{char_b}",
                    'char_a': char_a,
                    'char_b': char_b,
                    'order': order,
                    'p_div': p_div,
                    'p_conv': p_conv,
                    'transition_count': count_ab,
                    'total_a': total_a,
                    'total_b': total_b
                })

    df = pd.DataFrame(results)
    df = df[(df['p_div'] > 1e-5) | (df['p_conv'] > 1e-5)]
    # enlever les résultats pour le couple '%-#' (fin-début)
    df = df[~df['couple'].str.contains('%-#')]
    df = df[~df['couple'].str.contains('%-%')]
    df = df[~df['couple'].str.contains('#-#')]


    return df


############################ UTILS Analyze_vocab_size ####################################
def zipf_law_fit(flat_tokens):
    counts = Counter(flat_tokens)
    freqs = np.array(sorted(counts.values(), reverse=True))
    ranks = np.arange(1, len(freqs) + 1)

    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)

    slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_freqs)

    return slope, intercept, r_value**2

# Convertir les tokens en une longue chaîne de caractères
def corpus_to_string(chunks_df):
    return ''.join(
        ''.join(token) for row in chunks_df.values.flatten() if isinstance(row, list) for token in row
    )

# Convertir le nouveau token (tuple) en une chaîne de caractères
def tuple_to_string(token):
    return ''.join(token) if isinstance(token, tuple) else token

def count_token_occurrences(chunks_df):
    return Counter(
        token
        for row in chunks_df.values.flatten()
        if isinstance(row, list)
        for token in row
    )

def to_list_if_str(x):
    return str(x).split() if pd.notna(x) else []

def list_to_tuple(x):
    return tuple(x) if isinstance(x, list) else x

def list_chunk_lengths(x):
    return [len(chunk) for chunk in x] if isinstance(x, list) else []

def flatten_chunks_as_sequences(df):
    return set(tuple(row) for row in df.values.flatten() if isinstance(row, list))


def make_hashable_df(df):
    return df.apply(lambda row: row.map(list_to_tuple), axis=1)

def shannon_entropy(seq):
    if not seq:
        return 0.0
    total = len(seq)
    freqs = Counter(seq)
    return -sum((count/total) * math.log2(count/total) for count in freqs.values())

def entropies_from_labeled_sequence(Y):
    chants = []
    current = []

    for label in Y:
        if label == 'END':
            if current:
                chants.append(current)
                current = []
        else:
            current.append(label)
    if current:
        chants.append(current)

    return [shannon_entropy(chant) for chant in chants]



######################### Utils pour token presence in songs #################################

def extract_songs_from_cleaned_Y(Y_cleaned):
    """Extrait les chants individuels en utilisant # comme délimiteur"""
    songs = []
    current_song = []
    
    for label in Y_cleaned:
        if label == '#':
            if current_song:  # Si le chant actuel n'est pas vide
                songs.append(' '.join(current_song))  # Joindre en string pour comparaison avec tokens
            current_song = []
        elif label != '%':  # Ignorer les % (fins de chant)
            current_song.append(label)
    
    # Ajouter le dernier chant s'il existe
    if current_song:
        songs.append(' '.join(current_song))
    
    return [song for song in songs if song.strip()]  # Enlever les chants vides



def calculate_token_song_presence(tokens, songs):
    """Calcule dans combien de chants chaque token apparaît"""
    token_song_counts = {}
    
    for token in set(tokens):  # Pour chaque token unique
        songs_containing_token = 0
        for song in songs:
            if token in song:  # Vérifier si le token est présent dans le chant
                songs_containing_token += 1
        token_song_counts[token] = songs_containing_token
    
    return token_song_counts
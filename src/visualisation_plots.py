import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from collections import defaultdict

from src.detect_good_k_mean_alg import mean_cluster_hypersphere_density, detect_elbow_k
from src.utils import find_optimal_path_label_position, find_optimal_label_position, calculate_transition_probabilities

import numpy as np
from adjustText import adjust_text





############################# pConv/pDiv plot ######################################
def plot_convergence_divergence_clustered(
    df,
    order=1,
    path_groups=None,
    show_legend=True,
    use_clustering=True,
    show_path_legend=True,
    path_colors=None,
    path_width=2.5,
    arrow_size=15
):


    sns.set(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({'font.family': 'sans-serif'})

    # title = f"Transition probabilities - clustered (order {order})"
    title = "Comparison of chunks finding methods"
    X = df[['p_div', 'p_conv']].values

    if use_clustering:
        k_candidates = list(range(2, 12))
        densities = [mean_cluster_hypersphere_density(X, k) for k in k_candidates]
        optimal_k = detect_elbow_k(k_candidates, densities)
        print(f"Optimal number of clusters (k) selon hypersphere density elbow: {optimal_k}")

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X)
        df = df.copy()
        df['cluster'] = clusters
        palette = sns.color_palette("Set2", optimal_k)
    else:
        df = df.copy()
        df['cluster'] = 0
        palette = sns.color_palette("Set2", 1)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Dessiner les clusters
    if use_clustering:
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id]
            cluster_points = cluster_df[['p_div', 'p_conv']].values
            if len(cluster_points) >= 3:
                try:
                    hull = ConvexHull(cluster_points)
                    hull_vertices = cluster_points[hull.vertices]
                    polygon = plt.Polygon(
                        hull_vertices,
                        closed=True,
                        facecolor=palette[cluster_id],
                        edgecolor=palette[cluster_id],
                        alpha=0.15,
                        zorder=1
                    )
                    ax.add_patch(polygon)
                except:
                    pass

            ax.scatter(
                cluster_df['p_div'], cluster_df['p_conv'],
                s=80, alpha=0.85, edgecolor='white', linewidth=1.5,
                label=f"Cluster {cluster_id + 1}" if use_clustering else None,
                color=palette[cluster_id],
                zorder=8
            )
    else:
        ax.scatter(
            df['p_div'], df['p_conv'],
            s=80, alpha=0.85, edgecolor='white', linewidth=1.5,
            color=palette[0],
            zorder=8
        )
    # Collecter toutes les positions des labels de points
    point_label_positions = []
    
    for _, row in df.iterrows():
        x, y = row['p_div'], row['p_conv']
        label_x, label_y = find_optimal_label_position(x, y, point_label_positions)
        point_label_positions.append((label_x, label_y))
        
        ax.annotate(
            row['couple'],
            (x, y),
            xytext=(label_x, label_y),
            fontsize=9,
            alpha=0.9,
            weight='medium',
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor='white',
                alpha=0.85,
                edgecolor='lightgray',
                linewidth=0.8
            ),
            arrowprops=dict(
                arrowstyle='-',
                color='gray',
                alpha=0.6,
                linewidth=0.8
            ),
            zorder=15
        )

    if path_groups:
        all_drawn = set()
        colors = sns.color_palette("Accent", len(path_groups))
        shared_color = '#e74c3c'  # Rouge plus vif

        group_names = list(path_groups.keys())
        group_colors = dict(zip(group_names, colors))

        seen_paths = defaultdict(set)
        transition_to_groups = defaultdict(set)

        for group_name, paths in path_groups.items():
            for path in paths:
                seen_paths[path].add(group_name)
                for i in range(len(path) - order):
                    src = path[i]
                    tgt = path[i + order]
                    transition_to_groups[(src, tgt)].add(group_name)

        path_label_positions = []
        

        to_draw_shared = []
        to_draw_unique = []
        path_info_for_labels = []  # (start, end, path_string, color, is_shared)

        for group_name, paths in path_groups.items():
            for path_string in paths:
                if path_string in all_drawn:
                    continue

                pairs = [(path_string[i], path_string[i + order]) for i in range(len(path_string) - order)]
                coords = []
                missing = []

                for a, b in pairs:
                    row = df[df['couple'] == f"{a}-{b}"]
                    if not row.empty:
                        coords.append((row['p_div'].iloc[0], row['p_conv'].iloc[0]))
                    else:
                        missing.append(f"{a}-{b}")

                is_shared_path = len(seen_paths[path_string]) > 1

                if coords:
                    # Dessiner les segments
                    for i in range(len(coords) - 1):
                        src_token, tgt_token = pairs[i]
                        start, end = coords[i], coords[i + 1]
                        transition_groups = transition_to_groups[(src_token, tgt_token)]
                        
                        path_color = shared_color if is_shared_path else group_colors[group_name]

                        if len(transition_groups) > 1 and is_shared_path:
                            to_draw_shared.append((start, end))
                        else:
                            to_draw_unique.append((start, end, path_color))
                    
                    # Stocker les infos pour les labels
                    if len(coords) >= 2:
                        path_info_for_labels.append((
                            coords[-2], coords[-1], path_string, 
                            shared_color if is_shared_path else group_colors[group_name],
                            is_shared_path
                        ))

                    # Marqueurs début/fin de path
                    color = shared_color if is_shared_path else group_colors[group_name]
                    ax.scatter(*coords[0], marker='o', s=90,
                               facecolor=color, edgecolor='white', linewidth=1.5,
                               zorder=12, alpha=0.9)
                    ax.scatter(*coords[-1], marker='s', s=90,
                               facecolor=color, edgecolor='white', linewidth=1.5,
                               zorder=12, alpha=0.9)

                elif missing:
                    print(f"[info] Transitions absentes pour '{path_string}' : {missing}")

                all_drawn.add(path_string)

        # dessin des flèches avec des courbes
        arrow_counts = defaultdict(int)

        # Flèches uniques
        for start, end, color in to_draw_unique:
            key = (tuple(start), tuple(end))
            arrow_counts[key] += 1
            count = arrow_counts[key]

            # Courbure basée sur le nombre d'occurrences
            rad = 0 if count == 1 else 0.15 * count * (-1)**(count)
            connection_style = f"arc3,rad={rad:.2f}"

            ax.annotate(
                '', xy=end, xytext=start,
                arrowprops=dict(
                    arrowstyle='->',
                    color=color,
                    lw=path_width,
                    shrinkA=8, shrinkB=8,
                    connectionstyle=connection_style,
                    alpha=0.8
                ),
                zorder=5
            )

        # Flèches partagées
        for start, end in to_draw_shared:
            key = (tuple(start), tuple(end))
            arrow_counts[key] += 1
            count = arrow_counts[key]

            rad = 0 if count == 1 else 0.15 * count * (-1)**(count)
            connection_style = f"arc3,rad={rad:.2f}"

            ax.annotate(
                '', xy=end, xytext=start,
                arrowprops=dict(
                    arrowstyle='->',
                    color=shared_color,
                    lw=path_width + 0.5,
                    shrinkA=8, shrinkB=8,
                    connectionstyle=connection_style,
                    alpha=0.9
                ),
                zorder=6
            )

        # Placer les labels de chemins après les flèches 
        for start_coord, end_coord, path_string, color, is_shared in path_info_for_labels:
            side = -1 if "ZAB" in path_string else 1 if "TAB" in path_string else None
            label_x, label_y = find_optimal_path_label_position(
                start_coord, end_coord,
                path_label_positions + point_label_positions, 
                side=side
            )

            path_label_positions.append((label_x, label_y))
            
            ax.text(
                label_x, label_y, path_string,
                fontsize=10,
                color=color,
                weight='bold',
                ha='center', va='center',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor='white',
                    alpha=0.92,
                    edgecolor=color,
                    linewidth=1.8
                ),
                zorder=20
            )

        # Légende
        if show_legend:
            legend_elements = []
            for group_name, color in group_colors.items():
                legend_elements.append(
                    plt.Line2D([0], [0], color=color, linewidth=3, label=f"{group_name} sequences")
                )
            legend_elements.append(
                plt.Line2D([0], [0], color=shared_color, linewidth=3, label="Shared sequences")
            )
            
            if show_path_legend:
                from matplotlib.lines import Line2D
                legend_elements.extend([
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                           markersize=8, label="Path start"),
                    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                           markersize=8, label="Path end")
                ])
            
            ax.legend(handles=legend_elements, loc='upper right', 
                     frameon=True, fancybox=True, shadow=True)

    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('Divergence probability P(A → B)', fontsize=13)
    ax.set_ylabel('Convergence probability P(A ← B)', fontsize=13)
    
    if path_groups:
        ax.set_title(title + f"\nMethods compared : {', '.join(path_groups.keys())}", 
                    fontsize=14, weight='bold', pad=15)
    else:
        ax.set_title(title, fontsize=14, weight='bold', pad=15)
    
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Ligne de référence
    ax.plot([0, 1.2], [0, 1.2], '--', color='darkgray', alpha=0.5, 
           linewidth=1, zorder=0, label='P(div) = P(conv)')

    plt.tight_layout()

    plt.savefig(f"pConvpDiv_plot_order_{order}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"pConvpDiv_plot_order_{order}.pdf", format="pdf", bbox_inches='tight')

    plt.show()
    return fig






def analyze_sequence_probabilities(Y_clean, order=1, path_groups=None, verbose=False, seuil=50):


    
    df_probs = calculate_transition_probabilities(Y_clean, order=order, seuil=seuil)
    if verbose:
        print("Analyse des probabilités de transition...")
        print(f"Nombre total d'éléments dans la séquence: {len(Y_clean)}")
        
        print(f"Nombre de couples de caractères analysés: {len(df_probs)}")
        print("\nPremiers résultats:")
        print(df_probs.head(70).to_string(index=False))
        
        # Stats descriptives
        print("\nStatistiques des probabilités de divergence:")
        print(f"Moyenne: {df_probs['p_div'].mean():.4f}")
        print(f"Médiane: {df_probs['p_div'].median():.4f}")
        print(f"Max: {df_probs['p_div'].max():.4f}")
        
        print("\nStatistiques des probabilités de convergence:")
        print(f"Moyenne: {df_probs['p_conv'].mean():.4f}")
        print(f"Médiane: {df_probs['p_conv'].median():.4f}")
        print(f"Max: {df_probs['p_conv'].max():.4f}")
    
    plot_convergence_divergence_clustered(df_probs,  path_groups=path_groups, order=order, path_width=2, show_path_legend=False, use_clustering=True)
    
    return df_probs


############################################### Plot courbes de métriques et entropie ##################################


def plot_tokenization_analysis(results_df, new_token_occurrence_df, tokenizer_type,
                               x_vlines=(40, 47, 62, 81),
                               vline_colors=('#228B22', '#4169E1', '#FF4500', '#8B0000'),
                               output_prefix="token_analysis"):
    """
    Génère et sauvegarde une figure d'analyse de la tokenisation comprenant :
    1. Token occurrences par quantile
    2. Fréquence normalisée par longueur de token
    3. Entropie de Shannon de la distribution des tokens

    Parameters
    ----------
    results_df : pd.DataFrame
        Contient 'vocab_size', 'entropy' et les colonnes qXX_occurrences
    new_token_occurrence_df : pd.DataFrame
        Contient 'vocab_size', 'token', 'occurrence'
    tokenizer_type : str
        Nom ou type du tokenizer (utilisé dans le titre et nom de fichier)
    x_vlines : tuple
        Positions des lignes verticales de référence
    vline_colors : tuple
        Couleurs correspondantes aux lignes verticales
    output_prefix : str
        Préfixe des fichiers de sortie PNG et PDF
    """

    # Style publication scientifique
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern Roman'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'legend.title_fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'axes.linewidth': 1,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3
    })

    # Bornes X
    x_min = min(results_df['vocab_size'].min(), new_token_occurrence_df['vocab_size'].min()) - 1
    x_max = max(results_df['vocab_size'].max(), new_token_occurrence_df['vocab_size'].max()) + 1

    # Données supplémentaires
    new_token_occurrence_df = new_token_occurrence_df.copy()
    new_token_occurrence_df['token_length'] = new_token_occurrence_df['token'].apply(len)
    new_token_occurrence_df['occurrence_per_char'] = (
        new_token_occurrence_df['occurrence'] / np.sqrt(new_token_occurrence_df['token_length'])
    )

    # Figure et axes
    fig, axes = plt.subplots(3, 1, figsize=(12, 7))
    fig.patch.set_facecolor('white')
    vline_labels = [f'Vs={x}' for x in x_vlines]

    # --- Graphique 1 ---
    ax1 = axes[0]
    quantile_list = [25, 40, 50, 60, 75, 100]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for q, color in zip(quantile_list, colors):
        col_name = f'q{q}_occurrences'
        if col_name in results_df.columns:
            ax1.plot(
                results_df['vocab_size'],
                results_df[col_name],
                label=f'Q{q}',
                color=color,
                marker='o',
                linewidth=2,
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.5,
                markeredgecolor=color
            )

    for x, color in zip(x_vlines, vline_colors):
        ax1.axvline(x=x, color=color, linestyle='--', linewidth=1.5)

    ax1.set_title('Token Occurrences by Length Quantile', fontweight='bold', pad=15)
    ax1.set_xlabel('Vocabulary Size')
    ax1.set_ylabel('Token Occurrences')
    ax1.set_xlim(x_min, x_max)
    ax1.set_yscale('log')
    ax1.legend(
        title="Quantiles",
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True
    ).get_frame().set_facecolor('white')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)

    # --- Graphique 2 ---
    ax2 = axes[1]
    top_tokens = new_token_occurrence_df.nlargest(15, 'occurrence_per_char')
    ax2.plot(
        new_token_occurrence_df['vocab_size'], 
        new_token_occurrence_df['occurrence_per_char'], 
        marker='o',
        color='#DC143C',
        linewidth=2,
        markersize=5,
        markerfacecolor='white',
        markeredgewidth=1.5,
        markeredgecolor='#DC143C'
    )

    for x, color in zip(x_vlines, vline_colors):
        ax2.axvline(x=x, color=color, linestyle='--', linewidth=1.5)

    ax2.set_title('Token Frequency Normalized by Length', fontweight='bold', pad=15)
    ax2.set_xlabel('Vocabulary Size')
    ax2.set_ylabel('Occurrences / Length')
    ax2.set_xlim(x_min, x_max)

    for _, row in top_tokens.iterrows():
        ax2.annotate(
            row['token'],
            xy=(row['vocab_size'], row['occurrence_per_char']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'),
            arrowprops=dict(arrowstyle='->', color='gray')
        )

    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)

    # --- Graphique 3 ---
    ax3 = axes[2]
    ax3.plot(
        results_df['vocab_size'], 
        results_df['entropy'], 
        marker='o', 
        color='#228B22',
        linewidth=2,
        markersize=5,
        markerfacecolor='white',
        markeredgewidth=1.5,
        markeredgecolor='#228B22'
    )

    for x, color in zip(x_vlines, vline_colors):
        ax3.axvline(x=x, color=color, linestyle='--', linewidth=1.5)

    ax3.set_title('Shannon Entropy of Token Distribution', fontweight='bold', pad=15)
    ax3.set_xlabel('Vocabulary Size')
    ax3.set_ylabel('Entropy (bits)')
    ax3.set_xlim(x_min, x_max)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.set_axisbelow(True)

    # --- Titre et mise en page ---
    fig.suptitle(f'Tokenization Analysis: {tokenizer_type}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    vline_legend_elements = [
        plt.Line2D([0], [0], color=color, linestyle='--', linewidth=1.5, label=label)
        for color, label in zip(vline_colors, vline_labels)
    ]
    fig.legend(
        handles=vline_legend_elements,
        title="Reference Points",
        loc='lower right',
        bbox_to_anchor=(0.88, 0.02),
        frameon=True,
        fontsize=9,
        title_fontsize=10
    )

    # Sauvegarde
    png_path = f"{output_prefix}_{tokenizer_type}_publication.png"
    pdf_path = f"{output_prefix}_{tokenizer_type}_publication.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.show()



########################################## Plot Coverage of true chunks for 3 methods ###################################################

def count_token_presence_in_songs(Y_clean, token):
    """
    Compte dans combien de chants le token apparaît.
    
    Args:
        Y_clean: liste contenant tous les chants séparés par '#'
        token: le token à chercher (ex: 'ZAB')
    
    Returns:
        nombre de chants contenant ce token
    """
    # Séparer les chants (diviser par '#')
    songs = []
    current_song = []
    
    for label in Y_clean:
        if label == '#':
            if current_song:  # Si le chant n'est pas vide
                songs.append(current_song)
                current_song = []
        else:
            current_song.append(label)
    
    # Ajouter le dernier chant si il n'y a pas de '#' à la fin
    if current_song:
        songs.append(current_song)
    
    # Compter dans combien de chants le token apparaît
    count = 0
    token_sequence = list(token)  # Convertir 'ZAB' en ['Z', 'A', 'B']
    
    for song in songs:
        # Chercher la séquence dans le chant
        for i in range(len(song) - len(token_sequence) + 1):
            if song[i:i+len(token_sequence)] == token_sequence:
                count += 1
                break  # On compte une seule fois par chant
    
    return count

def filter_tokens_by_frequency(tokens, Y_clean, min_percentage=30):
    """
    Filtre les tokens selon leur pourcentage de présence dans les chants.
    
    Args:
        tokens: liste des tokens à filtrer
        Y_clean: liste contenant tous les chants séparés par '#'
        min_percentage: pourcentage minimum de présence (défaut: 30%)
    
    Returns:
        ensemble des tokens suffisamment fréquents
    """
    # Compter le nombre total de chants
    total_songs = Y_clean.count('#') + (1 if Y_clean and Y_clean[-1] != '#' else 0)
    
    # Filtrer les tokens
    frequent_tokens = set()
    for token in tokens:
        if token and token != "None":  # Ignorer les tokens vides ou None
            token_clean = token.replace(" ", "").replace("#", "")
            if token_clean and len(token_clean)>1:  # S'assurer que le token n'est pas vide après nettoyage et que c'est pas une seule phrase
                presence_count = count_token_presence_in_songs(Y_clean, token_clean)
                presence_percentage = (presence_count / total_songs) * 100
                
                if presence_percentage >= min_percentage:
                    frequent_tokens.add(token_clean)
    
    return frequent_tokens

def clean_chunks(chunk_list):
    """Nettoie la liste des chunks : supprime None, espaces, '#'."""
    return set(c.replace(" ", "").replace("#", "") 
               for c in chunk_list if c and c != "None")

def plot_chunk_distribution_filtered(bpe, wp, uni, true_chunks, Y_clean, min_percentage=30, horizontal=False):
    """
    Plots the distribution and coverage of "true chunks" (reference tokens) across three different tokenization methods
    (BPE, WordPiece, Unigram), after filtering tokens by their frequency of occurrence in the dataset.
    For each method, tokens are filtered to retain only those present in at least `min_percentage` percent of the songs.
    The plot shows how many of the true chunks are found by each method, how many other tokens are present, and which
    true chunks are missing from each method's vocabulary. The plot can be displayed either horizontally or vertically.
    Parameters
    ----------
    bpe : set or list
        Set or list of tokens from the BPE tokenization method.
    wp : set or list
        Set or list of tokens from the WordPiece tokenization method.
    uni : set or list
        Set or list of tokens from the Unigram tokenization method.
    true_chunks : iterable
        Collection of reference tokens ("true chunks") to check for coverage.
    Y_clean : list or array-like
        List of sequences (e.g., songs) used to compute token frequencies for filtering.
    min_percentage : int, optional
        Minimum percentage of songs in which a token must appear to be retained (default is 30).
    horizontal : bool, optional
        If True, plots horizontal bar chart; otherwise, plots vertical bar chart (default is False).
    Returns
    -------
    None
        The function displays and saves the plot as PNG and PDF files.
    """

    # Nettoyage des true_chunks
    true_chunks_set = set(true_chunks)
    
    # Filtrage par fréquence pour chaque méthode
    bpe_filtered = filter_tokens_by_frequency(bpe, Y_clean, min_percentage)
    wp_filtered = filter_tokens_by_frequency(wp, Y_clean, min_percentage)
    uni_filtered = filter_tokens_by_frequency(uni, Y_clean, min_percentage)
    
    # Récupération des données
    all_methods = [bpe_filtered, wp_filtered, uni_filtered]
    methods = ['BPE \nVs=69', 'WordPiece \nVs=62', 'Unigram \nVs=56']

    found_counts = []
    other_counts = []
    total_chunks = []
    missing_chunks = []

    for method_chunks in all_methods:
        found = true_chunks_set & method_chunks
        other = method_chunks - true_chunks_set
        missing = true_chunks_set - method_chunks

        found_counts.append(len(found))
        other_counts.append(len(other))
        total_chunks.append(len(method_chunks))
        missing_chunks.append(missing)



    if horizontal:
        fig, ax = plt.subplots(figsize=(10,6))
        # Barres horizontales
        bar1 = ax.barh(np.arange(len(methods)), found_counts, label='homemade chunks found', color='green')
        bar2 = ax.barh(np.arange(len(methods)), other_counts, left=found_counts, label='Other tokens', color='lightgray')

        ax.set_xlabel("Number of tokens", fontsize=18)
        ax.set_title(f"Coverage of true chunks \n(filtered by {min_percentage}% presence in songs)", fontsize=20)
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels(methods, fontsize=16)
        
        # Affichage des chunks manquants
        for i, missing in enumerate(missing_chunks):
            text = f"Missing: {', '.join(missing) if missing else 'None'}"
            ax.text(total_chunks[i] + 0.5, i, text, va='center', fontsize=14,
                    bbox=dict(facecolor='white', alpha=0.8))
        plt.xlim(0, max(total_chunks) + 5)

    else:
        # Barres verticales
        fig, ax = plt.subplots(figsize=(5.4,6))
        x = np.arange(len(methods))
        width = 0.8
        bar1 = ax.bar(x, found_counts, width, label='homemade chunks found', color='green')
        bar2 = ax.bar(x, other_counts, width, bottom=found_counts, label='Other tokens', color='lightgray')

        ax.set_ylabel("Number of tokens", fontsize=18)
        ax.set_title(f"Coverage of true chunks \n(filtered by {min_percentage}% presence in songs)", fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=16)

        for i, missing in enumerate(missing_chunks):
            text = f"Missing: {', '.join(missing) if missing else 'None'}"
            ax.text(x[i], total_chunks[i] + 0.5, text, ha='center', va='bottom', fontsize=14,
                    bbox=dict(facecolor='white', alpha=0.8))
        plt.ylim(0, max(total_chunks) + 5)

    ax.legend(fontsize=14)
    plt.tight_layout()
    if horizontal: 
        plt.savefig('chunk_coverage_filtered_horizontal.png', dpi=300, bbox_inches='tight')
        plt.savefig('chunk_coverage_filtered_horizontal.pdf', bbox_inches='tight')
    else : 
        plt.savefig('chunk_coverage_filtered_vertical.png', dpi=300, bbox_inches='tight')
        plt.savefig('chunk_coverage_filtered_vertical.pdf', bbox_inches='tight')
    plt.show()


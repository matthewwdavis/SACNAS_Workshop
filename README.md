sacnas_workshop
================
Matthew Davis
2025-05-16

## Table of Contents

- [Exploring Data](#explore)
- [Principal Components](#pcs)
- [Dendrograms](#dendrograms)
- [Conclusion](#conclusion)

## Exploring Data

### What is our goal?

In the `Sequences` directory, there are 10 files with the names
gene_a-j.Each one of these files is a DNA sequence of a specific gene
from a specific organism, and our job is to figure out what those genes
and organisms are. A good starting point is to use a tool from the
National Center for Biotechnology Information (NCBI) called Basic Local
Alignment Search Tool (BLAST).

### A quick refresher on BLAST

BLAST is an incredibly useful and cool tool. We covered how BLAST works
in a little more detail during the lecture, but as a quick reminder it
basically tells us how similar a sequence we provide (called the query)
is to the other sequences on NCBI. In 2023, NCBI GenBank had over 2.9
billion sequences for over half a million species, and that number has
only grown! I don’t think its hyperbole to say that most of the known
genetic sequence in the world is hosted on NCBI, and we can search
through it all.

### Using NCBI BLAST to identify our sequences

Talk about how to use blast Add in some screenshots so that they can
see. Make sure they are keeping track of their data! Please copy this
google sheet and fill it out

## Principal Components

### Load modules

Here are the modules we need to load so that we can plot our PCA. All of
these should be available in [Pickcode](https://pickcode.io/)! I also
tried to make sure I was commenting what the modules were being used
for. This can be really helpful when going back or sharing your code
with other scientists.

``` python
import pandas as pd # Used to create data tables
import numpy as np # Used for
from collections import Counter # Used for
from itertools import product # Used for
import matplotlib.pyplot as plt # Used for plotting
```

### Generating our data table

Please copy and paste this code chunk! This code is creating a data
table with the gene name and the first 500 nucleotides of DNA sequence
from each gene. We want to make sure that we are all using the same
sequences and that these sequences are correct. Normally, we would
download and load the data directly so that we can use the entire gene
sequence, but this should be okay for our purposes.

``` python
# Lets create a dictionary with two columns: "Gene" and "Sequence"
data = {
    "Gene": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    "Sequence": [
        "CGAGAACCCAGAAGACCTTTAATCTCCCGCCTCTTTCACACCATCAGACTCCATTATCGAAGCTCGTTCACTCTTCTCTCTCTCCAAATTCGAGAGAAAGGAGGAGGAAGGTGTAATGCCGCGTAAACCTAGACATCGTGTACCGGAGATTCTATGGAGGTTATTCGGAAACAGAGCCAGGAATTTAAACGACGCAATAGTAGATCTGATTCCTAACCGGAATATCCAGCCGGAGCAATGCCGATGCCGGGGTCAAGGTTGTCTCGGTTGCAGCAGCGATAAACCGGCGTTTCTGCTGCGTTCCGATGATCCCATTCACTACCGTAAACTTCTTCACCGTTGCTTCGTTGTACTTCACGAACAAACCCCTCCGCTTCTGGATTTCTCTCCAACATCTTGGTGGTCACAGAGAGAGATTGTTGAAAGGATTATTGAAATGATGCAATCTGGATGTGATTGCCAAAATGTGATATGTGCCAGATATGATAAGTATGATCAGT",
        "CTCTCCTCGCGGCGCGAGTTTCAGGCAGCGCTGCGTCCTGCTGCGCACGTGGGAAGCCCTGGCCCCGGCCACCCCCGCGATGCCGCGCGCTCCCCGCTGCCGAGCCGTGCGCTCCCTGCTGCGCAGCCACTACCGCGAGGTGCTGCCGCTGGCCACGTTCGTGCGGCGCCTGGGGCCCCAGGGCTGGCGGCTGGTGCAGCGCGGGGACCCGGCGGCTTTCCGCGCGCTGGTGGCCCAGTGCCTGGTGTGCGTGCCCTGGGACGCACGGCCGCCCCCCGCCGCCCCCTCCTTCCGCCAGGTGGGCCTCCCCGGGGTCGGCGTCCGGCTGGGGTTGAGGGCGGCCGGGGGGAACCAGCGACATGCGGAGAGCAGCGCAGGCGACTCAGGGCGCTTCCCCCGCAGGTGTCCTGCCTGAAGGAGCTGGTGGCCCGAGTGCTGCAGAGGCTGTGCGAGCGCGGCGCGAAGAACGTGCTGGCCTTCGGCTTCGCGCTGCTGGACGG",
        "GTTCCCAGCCTCATCTTTTTCGTCGTGGACTCTCAGTGGCCTGGGTCCTGGCTGTTTTCTAAGCACACCCTTGCATCTTGGTTCCCGCACGTGGGAGGCCCATCCCGGCCTTGAGCACAATGACCCGCGCTCCTCGTTGCCCCGCGGTGCGCTCTCTGCTGCGCAGCCGATACCGGGAGGTGTGGCCGCTGGCAACCTTTGTGCGGCGCCTGGGGCCCGAGGGCAGGCGGCTTGTGCAACCCGGGGACCCGAAGATCTACCGCACTTTGGTTGCCCAATGCCTAGTGTGCATGCACTGGGGCTCACAGCCTCCACCTGCCGACCTTTCCTTCCACCAGGTGGGCCTCCAGGCGGGATCCCCATGGGTCAGGGGCGGAAAGCCGGGAGGACGTGGGATAGTGCGTCTAGCTCATGTGTCAAGACCCTCTTCTCCTTACCAGGTGTCATCCCTGAAAGAGCTGGTGGCCAGGGTTGTGCAGAGACTCTGCGAGCGCAACGAG",
        "CCAATGCTTACACAGGCTAGAATCATCACAACCTTCATTTTTGTCCCAAATTATGATCTCAGATTGCCGTTGCTATAAATTTTGGAATTGTATTGGAAACTGTAACTGTGCGAAGGAGGGGATCTTTAATCCTAGTAAATCCAAAATCGGTTCACATAATCGACTTTTTCGTCATTGGATTTATTGGCTTTTTTCTTCTGTAGTGGTACCCGTTATCAGTAATTGTTTTTATGTTACTGAAAGGCAATTTGATAAGCTGCAAGTTTTTTATTATCCTAAACCTGTGTGGAAGATGTTAGCAGATAATGCCACCGTCTATTTGAAGGAACATAATTATGAACAATTGAATGCTGTGTCTTACATGTCTATAATTACGAAGAGGAAATTTGGTTTTTCAAGGGTGAGATTTTTACCAAAGAAAAATAAAATGCGGATAGTGGCAAATACTAAGGCACCATGTGAGATAAAAACTTCCGATCAAAGAAAGAAAAGTTTTTTTG",
        "GGCACGAGGCCAGCTCTGTCCCGAGCGCCCGTCCGTCCAGGGCTCTGCCGGCCGTCCGTCCAGGGCTCCGCCGGCCGCCTATCCGGGGGACCGCCGGCCTGCCAGCCGTCCGTCCAGGGGCAGCGCCGCCGGCCGCCCGTCCAGGGCACCACCACCTCGCCATCGACCGTCCATCCAGGTCACCCCGACGACGACGTGTGCTGCTACTGCTGATTCACTTTGAGGTGCACGATAGCTGGTGGAAGGGAAAGGCCAAAAGGGAGAAGAAAACGAGAAAATAAAAGTTTCAGGCGCCGCTTCAAAATCCCGCCTCAAACGGTCAAAAGCCCCTGATCCCCCTTTCCCCTACCGGGCCTCTCCGCTGCCGCCTCATCCTCCACCCCCCTCACTTTCCTGTCGCGATGCCACGGCGGCGGCGGCGGCGGCGTGCGGCACCCGGCGGCCAAGTTCCCCCAGAGCTACGCCTGGCTTATGGCGCCCGCGCGCTCACGCTCGGCCGC",
        "ATGGCTACGAGTAAGACGCGAGCGGGAATTCGACGCGAACGGAGGGCGCGAAAGCGACGCCAAGATGAAGAGGCGGCAGAGAATTGCGCGTTGGAGACCGATTCCACCATGCTTGAGTCGAAGTGCATGTCCGAAAAGCCGCTCACCGCGAACCCATTCATTTCGCACATCGACGTGAAGGAAGTCATTCGATTGAGCGACGTCTATCGTTGTTCACAAGATGTCTCAGTCGACGGGGATGGCGAGACTAGTGGACTGACGAGCGAGGAGGTGACTTTGTCGCCTGCCTCGACGGAATATCTGCAAACGTCGAGAAAGCGCTCGAAGTCGTACAGAAAGCCGAGTTGGCTTCGTAAGCGTGAAGCGCGGCTGAACTCAACGCATGCTCAGAGTTTGCGGACGCGAGAAGTGAAGCAAATGACCGCCGGCAACATCGTGAATCGACAAAAGAATGCGAGTTTGGGTAACGAGGTGAAACTTGCTGACGAGGTCTTCGATCG",
        "CCGTGGGGCCCGCTGCACGGCAGCGCTGCGTGCGGGGATGGAGCGCGGGGCTCAGCCGGGAGTCGGTGTGCGGCGGCTCCGCAATGTAGCGCGGGAGGAGCCCTTCGCCGCGGTCCTGGGCGCGCTGCGGGGCTGCTACGCCGAGGNCACGCCGCTGGAGGCCTTCGTCCGGCGGCTGCAGGAGGGTGGCACCGGGGAGGTCGAGGTGCTGCGAGGCGACGACGCTCAGTGCTACCGGACCTTCGTGTCGCAGTGCGTGGTGTGCGTCCCCCGCGGTGCTCGCGCCATCCCCCGGCCCATCTGCTTCCAGCAGTTATCCAGTCAGAGCGAAGTCATCACAAGAATCGTTCAGAGGCTGTGTGAAAAGAAAAAGAAGAACATCCTTGCGTATGGATACTCCTTGCTGGATGAGAACAGTTGTCACTTCAGAGTTTTGCCATCTTCGTGTATATACAGCTATCTGTCCAATACTGTAACAGAAACGATTCGCATCAGTGGCC",
        "AAATAGAACTCCCCAAGCACGCGCACAGATGTCTGGACAGTACTCGACAGATGGCGGATTTAGGCCGGTTTTGGAGATTCTGCGCTCCTTATATCCGGTCGTGCAGACTTTGGAGGAGTTCACCGACGGACTGCAATTCCCTGACGGCCGAAAGCCGGTTCTGCTGGAGGAAACAGACGGCGCGCGCTTTAAAAAGCTCCTCAGTGGACTTATTGTATGTGCGTACACGCCGCCGCAGCTGCGCGTCCCCGCCCAGCTCAGCACCCTGCCGGAGGTCTTGGCGTTCACTCTGAACCACATTAAACGTAAGAAACTGAGGAACGTCCTGGGCTTCGGTTATCAATGCAGCGACGTGACGACCAGTTCGGATCCCTTCCGTTTCCATGGCGACGTTTCGCAGACGGCTGCCTCCATCAGCACCAGCGAGGTCTGGAAGCGTATCAACCAGCGTCTGGGCACGGAGGTAACGCGGTACCTGCTGCAGGACTGTGCCGTTTTCA",
        "CTCATCTAGCAAAGCTGGACTTTAGCTTAAGGGTTGTTGTTTTAATCAAAATATAAATGATGGATAATTTTACTTTGCAAAGTTTAAAAAAAGATTTTGGCACATATTTTCAACAGTATTGTTTACATCATAAAATATTAATTAAAAAAAATAAATGCAATGCTTTTATAGTGTCTAACGTTTGTGATTTAAAAAAATGTATAGCTGCTGTTAATTGTCTCAAATTCAATATTAATAGGAAACTGAAGAAACTACAGACAGATGCCTTATTTATAAATTCTTTAAATCCTATTTATTATATACCAAAATTAGATGAACAATATAAAATTCAAGAAAAGATAAATAGGAATGATAACCAAGATAAACAGTATAAGATAAAAAAGAAGAAAAACTTCCATAAAGAAATAACGTATCACAATATTTTTCTTAATAATTTAAACTTAAATATTTTTTTATGTAATACATCTCAAATATTACCATATTATACTAATTACAAAAGT",
        "ATGAAAATCTTATTCGAGTTCATTCAAGACAAGCTTGACATTGATCTACAGACCAACAGTACTTACAAAGAAAATTTAAAATGTGGTCACTTCAATGGCCTCGATGAAATTCTAACTACGTGTTTCGCACTACCAAATTCAAGAAAAATAGCATTACCATGCCTTCCTGGTGACTTAAGCCACAAAGCAGTCATTGATCACTGCATCATTTACCTGTTGACGGGCGAATTATATAACAACGTACTAACATTTGGCTATAAAATAGCTAGAAATGAAGATGTCAACAATAGTCTTTTTTGCCATTCTGCAAATGTTAACGTTACGTTACTGAAAGGCGCTGCTTGGAAAATGTTCCACAGTTTGGTCGGTACATACGCATTCGTTGATTTATTGATCAATTATACAGTAATTCAATTTAATGGGCAGTTTTTCACTCAAATCGTGGGTAACAGATGTAACGAACCTCATCTGCCGCCCAAATGGGCTCAACGATCATCCTC"
 ]
}

# Next we will convert the dictionary into a pandas data frame
df = pd.DataFrame(data)

# Now lets view the data frame
print(df)
```

    ##   Gene                                           Sequence
    ## 0    A  CGAGAACCCAGAAGACCTTTAATCTCCCGCCTCTTTCACACCATCA...
    ## 1    B  CTCTCCTCGCGGCGCGAGTTTCAGGCAGCGCTGCGTCCTGCTGCGC...
    ## 2    C  GTTCCCAGCCTCATCTTTTTCGTCGTGGACTCTCAGTGGCCTGGGT...
    ## 3    D  CCAATGCTTACACAGGCTAGAATCATCACAACCTTCATTTTTGTCC...
    ## 4    E  GGCACGAGGCCAGCTCTGTCCCGAGCGCCCGTCCGTCCAGGGCTCT...
    ## 5    F  ATGGCTACGAGTAAGACGCGAGCGGGAATTCGACGCGAACGGAGGG...
    ## 6    G  CCGTGGGGCCCGCTGCACGGCAGCGCTGCGTGCGGGGATGGAGCGC...
    ## 7    H  AAATAGAACTCCCCAAGCACGCGCACAGATGTCTGGACAGTACTCG...
    ## 8    I  CTCATCTAGCAAAGCTGGACTTTAGCTTAAGGGTTGTTGTTTTAAT...
    ## 9    J  ATGAAAATCTTATTCGAGTTCATTCAAGACAAGCTTGACATTGATC...

### Counting Kmers

Now that we have our data frame of genes and sequences,

``` python
## Kmer analysis
# Define kmer functions
def all_possible_kmers(k):
    return [''.join(p) for p in product('ATCG', repeat=k)]

def get_kmer_counts(seq, k):
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    return Counter(kmers)

# Convert sequences to kmer count vectors
k = 3
kmer_list = all_possible_kmers(k)

def sequence_to_vector(seq):
    counts = get_kmer_counts(seq, k)
    return [counts.get(kmer, 0) for kmer in kmer_list]

# Create k-mer matrix
kmer_matrix = df["Sequence"].apply(sequence_to_vector)
X = np.vstack(kmer_matrix)
```

### Questions about Kmers

**Discussion is encouraged!**

1.  What kmer size did we use in this analysis?

2.  Is there a calculation you can think of that would tell you the
    number of possible nucleotide combinations for any kmer? Using that
    calculation, how many combinations are possible for a 6-mer?
    **Hint:** There are 4 nucleotides

3.  Assuming perfect linear scaling, if calculating all possible 3-mers
    takes 1 second, how long would it take to calculate all possible
    21-mers?

4.  Can you think of some reason you would want certain kmer sizes?
    Think about what the benefits and downsides of what using a larger
    kmer size might be. What about a smaller kmer size?

# Calculating principal components

``` python
# Center the data
X_meaned = X - np.mean(X, axis=0)

# Covariance matrix
cov_matrix = np.cov(X_meaned, rowvar=False)

# Eigen decomposition
eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

# Sort by largest eigenvalues
sorted_idx = np.argsort(eigen_values)[::-1]
sorted_eigenvalues = eigen_values[sorted_idx]
sorted_eigenvectors = eigen_vectors[:, sorted_idx]

# Project data to 2D
X_reduced = np.dot(X_meaned, sorted_eigenvectors[:, :5])

# Add results to DataFrame
df["PC1"] = X_reduced[:, 0]
df["PC2"] = X_reduced[:, 1]
df["PC3"] = X_reduced[:, 2]
df["PC4"] = X_reduced[:, 3]
df["PC5"] = X_reduced[:, 4]

# Variance explained
explained = sorted_eigenvalues / np.sum(sorted_eigenvalues)

print(df)
```

    ##   Gene                                           Sequence  ...        PC4        PC5
    ## 0    A  CGAGAACCCAGAAGACCTTTAATCTCCCGCCTCTTTCACACCATCA...  ...   6.558033  10.209760
    ## 1    B  CTCTCCTCGCGGCGCGAGTTTCAGGCAGCGCTGCGTCCTGCTGCGC...  ... -10.010074   1.157887
    ## 2    C  GTTCCCAGCCTCATCTTTTTCGTCGTGGACTCTCAGTGGCCTGGGT...  ...  -9.332557  -7.010812
    ## 3    D  CCAATGCTTACACAGGCTAGAATCATCACAACCTTCATTTTTGTCC...  ...  -9.921222 -10.056807
    ## 4    E  GGCACGAGGCCAGCTCTGTCCCGAGCGCCCGTCCGTCCAGGGCTCT...  ...  24.703474  -5.952950
    ## 5    F  ATGGCTACGAGTAAGACGCGAGCGGGAATTCGACGCGAACGGAGGG...  ...   5.949401  -5.935282
    ## 6    G  CCGTGGGGCCCGCTGCACGGCAGCGCTGCGTGCGGGGATGGAGCGC...  ...  -9.582262   4.388123
    ## 7    H  AAATAGAACTCCCCAAGCACGCGCACAGATGTCTGGACAGTACTCG...  ...  -0.725250  14.439617
    ## 8    I  CTCATCTAGCAAAGCTGGACTTTAGCTTAAGGGTTGTTGTTTTAAT...  ...   2.193913   7.165352
    ## 9    J  ATGAAAATCTTATTCGAGTTCATTCAAGACAAGCTTGACATTGATC...  ...   0.166546  -8.404887
    ## 
    ## [10 rows x 7 columns]

``` python
print(df.drop(df.columns[1], axis=1))
```

    ##   Gene        PC1        PC2        PC3        PC4        PC5
    ## 0    A -11.746344  14.057086  18.632635   6.558033  10.209760
    ## 1    B  54.029402 -15.850360 -12.422545 -10.010074   1.157887
    ## 2    C  24.513470  -5.146106  17.870816  -9.332557  -7.010812
    ## 3    D -40.144006  -2.292522   3.707498  -9.921222 -10.056807
    ## 4    E  33.170666 -18.370923   1.377500  24.703474  -5.952950
    ## 5    F   4.993250  35.503404 -17.691397   5.949401  -5.935282
    ## 6    G  24.534906   2.968586  -6.072828  -9.582262   4.388123
    ## 7    H  11.855237   6.303905   3.174442  -0.725250  14.439617
    ## 8    I -69.162152 -21.279637 -12.931740   2.193913   7.165352
    ## 9    J -32.044429   4.106569   4.355619   0.166546  -8.404887

# Plot principal components

## Create color map for plotting

``` python
color_map = {
    "A": "red",
    "B": "blue",
    "C": "green",
    "D": "yellow",
    "E": "purple",
    "F": "orange",
    "G": "pink",
    "H": "black",
    "I": "brown",
    "J": "silver",
}
```

# Plotting

``` python
# Plot PCA 
plt.figure(figsize=(7, 6))

for gene, group in df.groupby("Gene"):
    plt.scatter(group["PC1"], group["PC2"],
                label=gene,
                color=color_map.get(gene, "gray"))

plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
plt.title("PCA of Genes (k-mer counts)")
plt.legend()
plt.grid(False)
plt.show()
```

<img src="README_files/figure-gfm/unnamed-chunk-6-1.png" width="672" />

### Questions about the PCA

**Discussion is encouraged**

5.  In science, visualizations are critical for understanding.
    Currently, the PCA coloring is fairly useless. Change the color of
    points on the scatterplot to reflect the larger classifications of
    the organisms that you discovered in our initial exploration. What
    classifications did you choose, and what color is each
    classification?

**Bonus question:** Add in a column for classification to the data
frame. Color by that column instead of gene name.

6.  Now that the colors mean something, look back at what you learned
    from BLAST. Does the PCA make sense? Why or why not?

7.  Go back and re-run the analysis with several different kmer lengths.
    Take note of what is happening in the PCA as you change kmer size.
    You’ll need to look through the code and decide where we are setting
    the kmer length. **Hint:** your answer to question 1 should help
    you. **Important:** Try not to go higher than 6. You can if you
    want, but you may be waiting a while.

8.  What kmer sizes did you try? Did you notice anything different
    happening in the pca with larger kmers? What about with smaller
    kmers?

9.  For this use case, do you think a larger or smaller kmer size
    better? Is there a difference? If so, why do you think?

## Dendrograms

Explain dendrograms and how they are like phylogenies.

Before starting, lets reset the kmer length to the original value and
run all code up to this point again.

### Modules needed for creating phylogenies

Here are the modules we need to create the phylogenies. Again, I tried
to give quick comments on what we are using each for.

``` python
from scipy.spatial.distance import pdist, squareform # To calculate distance
from scipy.cluster.hierarchy import linkage, dendrogram # To build a phylogeny building
```

### Calculating distance and creating a matrix

Talk about calculating distance. What is euclidean distance and why the
matrix. Then the linkage matrix.

``` python
# Compute a pairwise distance matrix using euclidean distance
distances = pdist(X, metric='euclidean')
dist_matrix = squareform(distances)

# Generate linkage matrix (e.g., using UPGMA)
linkage_matrix = linkage(distances, method='average')
```

### Plotting the dendrogram

``` python
# Plot the dendrogram
plt.figure(figsize=(10, 6))
dend_data = dendrogram(linkage_matrix, labels=df["Gene"].tolist())
plt.title("Phylogeny based on k-mer counts")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
```

<img src="README_files/figure-gfm/unnamed-chunk-9-3.png" width="960" />

### Questions about dendrograms

1.  Based on what organisms you know the Genes are from, do these
    relationships make sense to you? Why or why not?

2.  Like in question 7, go back and re-run the analysis with several
    different kmer lengths. What kmer sizes did you try? Is anything
    changing in the dendrogram?

3.  For the dendrogram, do you think a larger or smaller kmer size
    better? Is there a difference? If so, why do you think?

**Bonus question:** Change the colors of the large clades in the
dengrogram to best reflect the larger scale classifications of the
organisms. Feel free to use whatever kmer length you think makes the
most sense. What colors did you choose and why?

## Conclusion

Now we’ve seen a brief intro on how we can use computer science to
understand biological relationships! This is just a tiny idea of some of
the things that scientist do, and you’ll see more in your next workshop
with Ethan. If you want to see more about bioinformatics, check out the
resources tab below.

## Resources

- R for data science

- R textbook

- BIS180L website

- Julins site with r tutorials

- Maybe this site. Preface that I have never used it, but it looks good
  (<https://github.com/ossu/bioinformatics>)

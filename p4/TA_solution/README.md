I have made some more changes recently to the code. If you have downloaded my code before 19th July
you might need to look through the changes that I did:

- line 155-156 some changes to the way clusters for K-means are printed (before this part of code was resulting in NullPointerException
in case some cluster was empty)
- method singleLinkageDistance line 245 for computing distance using Manhattan distance (previous version was using something similar to Euclidean)
- method findClusterForCountries (after for loop on lines 272-274 I had a statement Math.sqrt which was wrong, since this isn't Euclidean distance but Manhattan)
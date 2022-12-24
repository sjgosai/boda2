import sys

JOBS = 1000
print("\t".join(['--env CHUNK', '--env JOBS', '--env STEPS', '--env ADAPT', '--output OUTFILE']), file=sys.stdout)

output_str = 'gs://jax-tewhey-boda-project-data/contribution_scores/created_202212/scores/contrib_test_chunk_{}' + f'_of_{JOBS}.h5'

for i in range(JOBS):
	output = output_str.format(i+1)
	print('\t'.join([str(i), str(JOBS), '100', 'True', output]), file=sys.stdout)

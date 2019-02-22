import os
import pdb

missing_files = ''

a = file('testVidID.txt').readlines()
for idx, line in enumerate(a):
	line = line.strip()
	if os.path.exists('./test/'+line+'.mp4') == False:
		missing_files += line + '\n'

file('test_missing_fils.txt', 'w').write(missing_files)

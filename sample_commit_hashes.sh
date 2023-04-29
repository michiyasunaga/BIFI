cd BIFI/data/orig_bad_code
# The bad bad code is split into 5 chunks. So, we sample 4 datapoints from each sample.
shuf --random-source=orig.0.id -n 4 orig.0.id
shuf --random-source=orig.0.id -n 4 orig.1.id
shuf --random-source=orig.0.id -n 4 orig.2.id
shuf --random-source=orig.0.id -n 4 orig.3.id
shuf --random-source=orig.0.id -n 4 orig.4.id

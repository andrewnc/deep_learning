In this lab I used the BASICLSTMCell provided in Tensorflow to initially get my set up working, I then built the GRU cell and trained by char-rnn to generate taylor swift lyrics. These can be found in output.txt. Training step information can be found in output_info.txt

The network worked very well on the Alma dataset, as can be seen in alma_out.txt. However, it seems to have been less effective in both the GRU and BasicLSTM case on the taylor_swift dataset. I believe this is because of the larger vocab size and larger variety of structure in the data. 

In general this char-rnn was successful. 
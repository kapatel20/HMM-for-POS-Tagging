import json
import copy
import sys
import math


# #### Reading necessary files

train_file = open("data/train","r")
train_text = train_file.readlines()

dev_file = open("data/dev","r")
dev_text = dev_file.readlines()

test_file = open("data/test","r")
test_text = test_file.readlines()

# ## Task - 1

# #### writeVocab --> function to print vocabulary in vocab.txt file

def writeVocab(vocab):
    file = open("vocab.txt","w")
    file.write("1\t"+"<unk>\t"+str(vocab["<unk>"])+"\n")
    cnt=1
    for i in vocab:
        if i=="<unk>":
            continue
        cnt+=1
        file.write(str(cnt)+"\t"+i+"\t"+str(vocab[i])+"\n")
    file.close()


# #### Creating python dictionary from training data

vocab_dict = {}

for word in train_text:
    if word!="\n":
        w = word.split("\t")[1]
        if w not in vocab_dict:
            vocab_dict[w]=1
        else:
            vocab_dict[w]+=1


# #### Removing words with frequency less than threshold

filler_filter = 2
final_vocab_list = {"<unk>":0}
filler_words = []

for word in vocab_dict:
    if vocab_dict[word] < filler_filter:
        final_vocab_list["<unk>"]+=vocab_dict[word]
        filler_words.append(word)
    else:
        final_vocab_list[word] = vocab_dict[word]

final_vocab = sorted(final_vocab_list.items(),key=lambda x:x[1],reverse=True)
final_vocab = dict(final_vocab)
print("Vocabulary size is ", len(final_vocab))
print("Occurences of unknown words is ", final_vocab["<unk>"])
writeVocab(final_vocab)


# ## Task - 2

# #### Creating a dictionary to store Tags and their occurences in training data

tag_count_list = {}
for word in train_text:
    if word != "\n":
        tag = word.split("\t")[2]
        tag = tag.split("\n")[0]
        if tag not in tag_count_list:
            tag_count_list[tag]=1
        else:
            tag_count_list[tag]+=1


# #### Creating dictionary to store state transition occurencies

tag_transition_dict = {}
tag_sum = 0
for i in range(len(train_text)-1):
    if train_text[i]!="\n":
        if str(train_text[i].split("\t")[0]) =="1":
            first_tag = train_text[i].split("\t")[2].split("\n")[0]
            tag_sum+=1
            if (first_tag) not in tag_transition_dict:
                tag_transition_dict[(first_tag)] = 1
                
            else:
                tag_transition_dict[(first_tag)] += 1
            
        if train_text[i+1]!="\n":
            tag1 = train_text[i].split("\t")[2].split("\n")[0]
            tag2 = train_text[i+1].split("\t")[2].split("\n")[0]
            if (tag1,tag2) not in tag_transition_dict:
                tag_transition_dict[(tag1,tag2)] = 1
            else:
                tag_transition_dict[(tag1,tag2)] +=1


# #### Creating a dictionary for (Tag, Word) Occurences

tag_word_emission_dict = {}

for i in range(len(train_text)):
    if train_text[i]!="\n":
        w = train_text[i].split("\t")[1]
        t = train_text[i].split("\t")[2].split("\n")[0]
        if t not in tag_word_emission_dict:
            tag_word_emission_dict[t] = {w:1}
        else:
            if w not in tag_word_emission_dict[t]:
                tag_word_emission_dict[t][w] = 1
            else:
                tag_word_emission_dict[t][w] += 1

tag_word_emission_dict_copy = copy.deepcopy(tag_word_emission_dict)
for tag in tag_word_emission_dict:
    tag_word_emission_dict_copy[tag]["<unk>"]=0
    for word in tag_word_emission_dict[tag]:
        if vocab_dict[word] < filler_filter:
            tag_word_emission_dict_copy[tag]["<unk>"] += tag_word_emission_dict_copy[tag][word]
            del tag_word_emission_dict_copy[tag][word]
tag_word_emission_dict = copy.deepcopy(tag_word_emission_dict_copy)


# #### Creating tag transition probabilities

transition_prob_dict = {}

for tags in tag_transition_dict:
    if type(tags) != str:
        tag1 = tags[0]
        tag2 = tags[1]
        transition_prob_dict[(tag1,tag2)] = tag_transition_dict[(tag1,tag2)]/tag_count_list[tag1]
    else:
        transition_prob_dict[tags] = tag_transition_dict[tags]/tag_sum


# #### Creating emission probablities

emission_prob_dict = {}

for tag in tag_word_emission_dict:
    for word in tag_word_emission_dict[tag]:
        emission_prob_dict[(tag,word)] = tag_word_emission_dict[tag][word]/tag_count_list[tag]


# #### Creating JSON file for transition and emission probabilites

hmm_dict = {
    "transition":str(transition_prob_dict),
    "emission":str(emission_prob_dict)
}

hmm = json.dumps(hmm_dict)


model_file = open("hmm.json",'w')
model_file.write(hmm)


# ## Task - 3

# #### Function for greedy decoding from HMM Probabilities

smoothing_prob = 0.00000001
def greedyDecoding(word, prevState):
    if prevState == "INIT":
        max_val = -1000
        state = ""
        
        for tag in list(tag_count_list.keys()):
            temp_val=-1
            try:
                temp_val = transition_prob_dict[tag]
            except KeyError:
                temp_val = 0
            try:
                temp_val *= emission_prob_dict[(tag,word)]
            except KeyError:
                temp_val *= smoothing_prob
            temp_val += smoothing_prob
            if temp_val>max_val:
                max_val = temp_val
                state = tag
        return state
    max_val = -1000
    state = ""
    
    for tag in list(tag_count_list.keys()):
            
        temp_val=-1
        
        try:
            temp_val = transition_prob_dict[(prevState, tag)]
        except KeyError:
            temp_val = 0
        try:
            temp_val *= emission_prob_dict[(tag,word)]
        except KeyError:
            temp_val *= smoothing_prob
        temp_val += smoothing_prob
        if temp_val>max_val:
            max_val = temp_val
            state = tag
    return state


# #### Function to create .out file for test data

def writeOutput(output_tags, algo):
    out_file = open(algo+".out","w")
    for i in output_tags:
        out_file.write(i)
    out_file.close()


# #### Function for getting POS tags for dev data using greedy decoding

def POS_Tags(file):
    final_POS_Tags = []
    cnt_total = 0
    cnt_wrong = 0
    flag = True
    prevState = "INIT"
    word_list = []
    for word in file:
        if word=="\n":
            final_POS_Tags.append("\n")
            continue
        temp=[]
        ind = word.split("\t")[0]
        w = word.split("\t")[1]
        t = word.split("\t")[2].split("\n")[0]
 
        if not flag and ind=="1":
            final_POS_Tags.append("\n")
            flag=True
            prevState = "INIT"
        
        pred = greedyDecoding(w,prevState)
        
        if pred != t:
            cnt_wrong+=1
        cnt_total+=1
        flag=False
        prevState = pred
        temp.append(ind+"\t"+w+"\t"+pred+"\n")
        final_POS_Tags.extend(temp)

    print("Accuracy on Dev Data for Greedy Algorithm : ", (1-cnt_wrong/cnt_total))
    acc = (1-cnt_wrong/cnt_total)
    return acc,final_POS_Tags            


# #### Function for getting POS tags for test data using greedy decoding

def POS_Tags_Test(file):
    final_POS_Tags = []
    flag = True
    prevState = "INIT"
    word_list = []
    for word in file:
        if word=="\n":
            final_POS_Tags.append("\n")
            continue
        temp=[]
        ind = word.split("\t")[0]
        w = word.split("\t")[1].split("\n")[0]
        if not flag and ind=="1":
            flag=True
            prevState = "INIT"
        
        pred = greedyDecoding(w,prevState)
        flag=False
        prevState = pred
        temp.append(ind+"\t"+w+"\t"+pred+"\n")
        final_POS_Tags.extend(temp)

    return final_POS_Tags            

dev_accuracy_greedy, final_greedy_tags = POS_Tags(dev_text)

final_greedy_test_tags = POS_Tags_Test(test_text)

writeOutput(final_greedy_test_tags,"greedy")


# ## Task - 4

tag_viterbi = list(tag_count_list.keys())
ind = [*range(0,len(tag_viterbi),1)]

tag_ind = dict(zip(ind,tag_viterbi))
# print(tag_ind)


# #### Function for Viterbi decoding on HMM Probabilites

viterbi_smoothing_prob = smoothing_prob
def viterbiDecoding(sentence):
    word_ind = [*range(0,len(sentence),1)]
    word_ind = dict(zip(word_ind,sentence))
    
    dp = [[0 for i in range(len(tag_viterbi))] for j in range(len(sentence))]
    dp_backtrack = [[-1 for i in range(len(tag_viterbi))] for j in range(len(sentence))]
    
    for tag in range(len(tag_viterbi)):
        try:
            viterbi_tag_prob = transition_prob_dict[tag_ind[tag]]
        except KeyError:
            viterbi_tag_prob = 0
            continue
        try:
            viterbi_emission_prob = emission_prob_dict[(tag_ind[tag],word_ind[0])]
        except KeyError:
            viterbi_emission_prob = viterbi_smoothing_prob
        dp[0][tag] = (viterbi_tag_prob*viterbi_emission_prob)

    for w in range(1,len(sentence)):
        for t in range(len(tag_viterbi)):
            temp_prob = -1
            try:
                viterbi_emission_prob = emission_prob_dict[(tag_ind[t],word_ind[w])]
            except KeyError:
                viterbi_emission_prob = viterbi_smoothing_prob
            best_path = 0
            for prev_t in range(len(tag_viterbi)):
                if dp[w-1][prev_t]==0:
                    continue
                try:
                    viterbi_tag_transition = transition_prob_dict[(tag_ind[prev_t],tag_ind[t])]
                except KeyError:
                    viterbi_tag_transition = 0
                    continue
                prob = (dp[w-1][prev_t]*viterbi_tag_transition*viterbi_emission_prob)
                if temp_prob<prob:
                    temp_prob = prob
                    best_path = prev_t
            dp[w][t] = temp_prob
            dp_backtrack[w][t] = best_path

    answers = []
    val = -1000
    start_index = -1
    for i in range(len(tag_viterbi)):
        if dp[-1][i] > val:
            start_index = i
            val = dp[-1][i]
        
    
    answers.append(start_index)
    for i in range(len(sentence)-1,0,-1):
        prev_max = dp_backtrack[i][start_index]
        answers.append(prev_max)
        start_index = prev_max
    
    final_tags = []
    
    for i in range(len(answers)-1,-1,-1):
        final_tags.append(tag_ind[answers[i]])
    return final_tags
    
# #### Function for getting POS tags for dev data using viterbi decoding

def POS_Tagging_Viterbi(file):
    final_tagging = []
    temp = []
    ind_list =[]
    true_tags = []
    cnt_wrong = 0
    cnt = 0
    final_viterbi_tags = []
    
    for word in file:
        if word=="\n":
            viterbi_tags = viterbiDecoding(temp)
            for i in range(len(viterbi_tags)):
                if viterbi_tags[i]!=true_tags[i]:
                    cnt_wrong+=1
            
            temp_answer = []
            for i in range(len(temp)):
                temp_answer.append(str(ind_list[i])+"\t"+temp[i]+"\t"+viterbi_tags[i]+"\n")
            final_viterbi_tags.extend(temp_answer)
            cnt+=len(viterbi_tags)
            true_tags=[]
            temp=[]
            ind_list = []
            final_viterbi_tags.append("\n")
            continue
        i_ind = word.split("\t")[0]
        w = word.split("\t")[1]
        t = word.split("\t")[2].split("\n")[0]
        temp.append(w)
        true_tags.append(t)
        ind_list.append(i_ind)
    viterbi_tags = viterbiDecoding(temp)
            
    temp_answer = []
    for i in range(len(temp)):
        temp_answer.append(str(ind_list[i])+"\t"+temp[i]+"\t"+viterbi_tags[i]+"\n")
    final_viterbi_tags.extend(temp_answer)
    acc = (1-(cnt_wrong/cnt))
    print("Accuracy on Dev Data for Viterbi Algorithm : ",acc)
    return acc, final_viterbi_tags


# #### Function for getting POS tags for test data using viterbi decoding

def POS_Tagging_Viterbi_Test(file):
    final_tagging = []
    temp = []
    ind_list =[]
    true_tags = []
    cnt_wrong = 0
    cnt = 0
    final_viterbi_tags = []
    
    for word in file:
        if word=="\n":
            viterbi_tags = viterbiDecoding(temp)
            temp_answer = []
            for i in range(len(temp)):
                temp_answer.append(str(ind_list[i])+"\t"+temp[i]+"\t"+viterbi_tags[i]+"\n")
            final_viterbi_tags.extend(temp_answer)
            cnt+=len(viterbi_tags)
            true_tags=[]
            temp=[]
            ind_list = []
            final_viterbi_tags.append("\n")
            continue
        i_ind = word.split("\t")[0]
        w = word.split("\t")[1].split("\n")[0]
        temp.append(w)
        ind_list.append(i_ind)
    viterbi_tags = viterbiDecoding(temp)
            
    temp_answer = []
    for i in range(len(temp)):
        temp_answer.append(str(ind_list[i])+"\t"+temp[i]+"\t"+viterbi_tags[i]+"\n")
    final_viterbi_tags.extend(temp_answer)
    return final_viterbi_tags

viterbi_acc, final_dev_viterbi_tags = POS_Tagging_Viterbi(dev_text)

# print(viterbi_acc)

final_test_viterbi_tags = POS_Tagging_Viterbi_Test(test_text)

writeOutput(final_test_viterbi_tags,"viterbi")





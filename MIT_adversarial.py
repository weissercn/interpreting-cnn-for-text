import json
from data import load_data, get_epoch
import torch
import math
from torch import nn
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

def run_adversary_attack(model, data, config, advers_attack):

    model.eval()
    n_iter = 0
    #epoch_x, epoch_y, lengths_x = get_epoch(data["valid_x"], data["valid_y"], config["batch_size"], is_train=False)
    epoch_x, epoch_y, lengths_x = get_epoch(data["valid_x"], data["valid_y"], 1, is_train=False) #num_examples=1)
    epoch_loss = 0
    corrects = 0
    criterion = nn.CrossEntropyLoss()


    results = {    
        "TP": [0,0], # number of examples changed and unchanged by adversarial attack
        "TN": [0,0],
        "FP": [0,0],
        "FN": [0,0], 
        }

    #print(len(epoch_x) , " examples")

    results_extended = []

    for batch_x, batch_y, length_x in zip(epoch_x, epoch_y, lengths_x):
        #batch_x_advers = [a+advers_attack for a in batch_x ]
        #length_x_advers = [a+len(advers_attack) for a in  length_x]
        
        batch_x_orig = batch_x.copy()
        length_x_orig = length_x.copy()
        #batch_x_advers_orig = batch_x_advers.copy()

        batch_x = torch.LongTensor(batch_x)
        batch_y = torch.LongTensor(batch_y)
        lengths_x = torch.LongTensor(length_x)

        if config["cuda"]:
            batch_x, batch_y, lengths_x = batch_x.cuda(), batch_y.cuda(), lengths_x.cuda()

        # optimizer.zero_grad()
        pred = model(batch_x)['logits']
        pred_class = torch.max(pred, 1)[1].view(batch_y.size()).data
        batch_y_orig = batch_y.clone()
        if config["cuda"]:
            pred_class = pred_class.cpu().detach().numpy()
            batch_y = batch_y.cpu().detach().numpy()

        #this only works if batch size is 1

        TYPE = ""
        if pred_class[0]==1 and batch_y[0]==1: TYPE = "TP"
        elif pred_class[0]==0 and batch_y[0]==0: TYPE = "TN"
        elif pred_class[0]==1 and batch_y[0]==0: TYPE = "FP"
        elif pred_class[0]==0 and batch_y[0]==1: TYPE = "FN"

        #if TYPE=="TN": # Adversarial attack on negative samples. If the model correctly predicts that a sample belongs to class 0. 
            # TN
        if True: 
            batch_x_advers_orig = [a+advers_attack[TYPE] for a in batch_x_orig ]
            length_x_advers_orig = [a+len(advers_attack[TYPE]) for a in  length_x_orig]
            
            batch_x_advers = torch.LongTensor(batch_x_advers_orig)
            lengths_x_advers = torch.LongTensor(length_x_advers_orig)

            if config["cuda"]:
                batch_x_advers, lengths_x_advers = batch_x_advers.cuda(), lengths_x_advers.cuda()

            # optimizer.zero_grad()
            pred_advers = model(batch_x_advers)['logits']
            pred_class_advers = torch.max(pred_advers, 1)[1].view(batch_y_orig.size()).data
            if config["cuda"]:
                pred_class_advers = pred_class_advers.cpu().detach().numpy()

            if pred_class[0] == pred_class_advers[0]: results[TYPE][1]+=1
            else: results[TYPE][0]+=1

            results_extended.append([batch_y[0], pred_class[0],pred_class_advers[0], length_x_orig[0]  ])

            if False:
                #print(data["idx_to_word"].keys(), len(data["idx_to_word"].keys()))
                print("Original sentence : ", " ".join([data["idx_to_word"][a] for a in batch_x_orig[0]  ]))
                print("Adversarial sentence : ", " ".join([data["idx_to_word"][a] for a in batch_x_advers_orig[0]  ]))
                print("Truth {}, Pred {}, Adversarial {}".format(batch_y[0],pred_class[0],pred_class_advers[0] ))


    return results, results_extended
    

def plot_adversarial_conversions(labels, means1, means2, fname, TPBool):
    
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, means1, width, label='natural')
    rects2 = ax.bar(x + width/2, means2, width, label='synthetic')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if TPBool:
        ax.set_ylabel('Fraction of TN converted')
        ax.set_title('TN converted by adversarial attack')
    else:
        ax.set_ylabel('Fraction of TP converted')
        ax.set_title('TP converted by adversarial attack')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    #autolabel(rects1)
    #autolabel(rects2)

    fig.tight_layout()

    plt.savefig(fname)

def plot_pred_advers_len(pred_advers_len, fname, title):
    data_1 = [length for ptype, length in pred_advers_len if ptype==0]
    data_2 = [length for ptype, length in pred_advers_len if ptype==1]

    print("data_1 ", data_1)

    bins = np.linspace(0, 50, 11)

    plt.title(title)
    plt.hist(data_1, bins, alpha=0.5, label="unchanged by attack")
    plt.hist(data_2, bins, alpha=0.5, label="flipped by attack")
    plt.legend()
    plt.xlabel("Words in Sentence")
    plt.ylabel("Number of Examples")
    plt.savefig(fname)

def eval_adversary(model, data, config):

    #Natural Ngrams
 
    # TN
    # f0 "cho", "]" 12.7  # 'TN': [93, 685]

    filter_maximising_ngrams = {
        "w2.f0": [["cho ]"] , ["surreal heart"]],
        "w2.f1": [["a delicate"], ["reflect intimacy"]], #natural attack does better
        "w2.f2": [["otherwise talented"], ["unique poorly"]],
        "w2.f3": [["daytime soap"], ["animal relative"]],
        "w2.f4": [["jessica stein"], ["timely martha"]],
        "w2.f5": [["spider web"], ["everyday russian"]],
        "w2.f6": [["quaid ,"], ["secretary films"]],
        "w2.f7": [["poorly staged"], ["boring mined"]],
        "w2.f8": [["two features"], ["terrifying warm"]],
        "w2.f9": [["according to"], ["forever player"]],
        "w2.f10": [['" 13'], ["jealousy explores"]],
        "w2.f11": [["plain boring"], ["exercise boring"]],
        "w2.f12": [["drag how"], ["immediately jokes"]],
        "w2.f13": [["generic effort"], ["generic thing"]],
        "w2.f14": [["major problem"], ["uneasy ridiculous"]], #natural attack does better
        "w2.f15": [["korean cinema"], ["pushing cinema"]], #natural attack does better
        "w2.f16": [["was supposed"], ["wasted listless"]], 
        "w2.f17": [["boring before"], ["boring momentos"]],
        "w2.f18": [["and heartwarming"], ["delivers heartwarming"]],
        "w2.f19": [["and manages"], ["below manages"]], #natural attack does better
    }

    """
    filter_maximising_ngrams = {
        "w2.f2": [["otherwise talented"], ["unique poorly"]],
    }
    """

    pos_filters_known =  ['w2.f0', 'w2.f1', 'w2.f4', 'w2.f5', 'w2.f6', 'w2.f8', 'w2.f10', 'w2.f15', 'w2.f18', 'w2.f19']
    neg_filters_known =  ['w2.f2', 'w2.f3', 'w2.f7', 'w2.f9', 'w2.f11', 'w2.f12', 'w2.f13', 'w2.f14', 'w2.f16', 'w2.f17']


    if False:
        pos_filters = []
        neg_filters = []
        all_filters = []

        pos_results = []
        neg_results = []
        all_results = []

        for k in filter_maximising_ngrams:
            advers_attack = {
            "TP": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][0][0].split(" ")],
            "TN": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][0][0].split(" ")],
            "FP": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][0][0].split(" ")],
            "FN": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][0][0].split(" ")],
            }
            #Natural
            results_natural, results_natural_extended = run_adversary_attack(model=model, data=data, config=config, advers_attack=advers_attack)
            hint_pos_natural = results_natural["TN"][0]*1./(results_natural["TN"][0]+ results_natural["TN"][1])
            hint_neg_natural = results_natural["TP"][0]*1./(results_natural["TP"][0]+ results_natural["TP"][1])
            print("{} {:.1e} {:.1e}".format(k, hint_pos_natural, hint_neg_natural))

            advers_attack = {
            "TP": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][1][0].split(" ")],
            "TN": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][1][0].split(" ")],
            "FP": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][1][0].split(" ")],
            "FN": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][1][0].split(" ")],
            }

            results_synthetic, results_synthetic_extended = run_adversary_attack(model=model, data=data, config=config, advers_attack=advers_attack)
            hint_pos_synthetic = results_synthetic["TN"][0]*1./(results_synthetic["TN"][0]+ results_synthetic["TN"][1])
            hint_neg_synthetic = results_synthetic["TP"][0]*1./(results_synthetic["TP"][0]+ results_synthetic["TP"][1])
            print("{} {:.1e} {:.1e}".format(k, hint_pos_synthetic, hint_neg_synthetic)) 

            if hint_pos_synthetic > hint_neg_synthetic: 
                pos_filters.append(k)
                pos_results.append([hint_pos_natural, hint_pos_synthetic])
            else: 
                neg_filters.append(k)
                neg_results.append([hint_neg_natural, hint_neg_synthetic])


            print("results_natural_extended ", results_natural_extended) 

            all_filters.append(k)
            all_results.append([hint_pos_natural, hint_pos_synthetic, hint_neg_natural, hint_neg_synthetic])

        print("pos_filters ", pos_filters)
        print("neg_filters ", neg_filters)

        plot_adversarial_conversions(pos_filters, [pos_result[0] for pos_result  in pos_results], [pos_result[1] for pos_result  in pos_results], "pos_adversarial_conversions.png", TPBool=True)
        plot_adversarial_conversions(neg_filters, [neg_result[0] for neg_result  in neg_results], [neg_result[1] for neg_result  in neg_results], "neg_adversarial_conversions.png", TPBool=False)

        plot_adversarial_conversions(all_filters, [all_result[0] for all_result  in all_results], [all_result[1] for all_result  in all_results], "all_pos_adversarial_conversions.png", TPBool=True)
        plot_adversarial_conversions(all_filters, [all_result[2] for all_result  in all_results], [all_result[3] for all_result  in all_results], "all_neg_adversarial_conversions.png", TPBool=False)

    if True:
        ## CHECK DEPENDENCE ON PHRASE LENGTH
        k = "w2.f6"        
        advers_attack = {
        "TP": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][0][0].split(" ")],
        "TN": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][0][0].split(" ")],
        "FP": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][0][0].split(" ")],
        "FN": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][0][0].split(" ")],
        }
        #Natural
        results_natural, results_natural_extended = run_adversary_attack(model=model, data=data, config=config, advers_attack=advers_attack)
        pred_advers_len = [ (pred_class_advers, length_x_orig)  for batch_y, pred_class, pred_class_advers, length_x_orig in results_natural_extended if (batch_y==0) and (pred_class==0) ]
    
        plot_pred_advers_len(pred_advers_len, "pos_len_dependence_natural.png", "Natural Attack on TN")



        """
        advers_attack = {
        "TP": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][1][0].split(" ")],
        "TN": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][1][0].split(" ")],
        "FP": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][1][0].split(" ")],
        "FN": [data["word_to_idx"][a] for a in filter_maximising_ngrams[k][1][0].split(" ")],
        }

        results_synthetic, results_synthetic_extended = run_adversary_attack(model=model, data=data, config=config, advers_attack=advers_attack)

        k = "w2.f16"
        """


if __name__ == '__main__': #original
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=True)

    args = parser.parse_args()

    with open(args.config) as fp:
        adversarial_config = json.load(fp)

    model_path = adversarial_config["model_path"]
    with open(model_path+'/config.json') as fp:
        config = json.load(fp)

    config.update(adversarial_config)

    with open(model_path+'/w2i.json') as fp:
        w2i = json.load(fp)

    data = load_data(config=config, word_to_idx=w2i)


    if config["cuda"]:
        model = torch.load(model_path+'/model')
        model = model.cuda()
    else:
        model = torch.load(model_path+'/model', map_location=torch.device('cpu'))


    eval_adversary(model=model, data=data, config=config)



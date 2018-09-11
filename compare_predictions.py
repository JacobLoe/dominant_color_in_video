#####################################################
## libraries
#####################################################
import extract_dominant_color as edc
#####################################################
## functions
#####################################################
def color_accuracy(colors_target,colors_predictions,*thing):
    acc=[]
    tl=0
    pl=0
    if thing:
        for target,pred in zip(colors_target,colors_predictions):
            acc.append(len(list(set(target) & set(pred)))/len(target))
        return acc
    for target,pred in zip(colors_target,colors_predictions):
        tl+=len(target)
        pl+=len(list(set(target) & set(pred)))
    return pl/tl
#####################################################
def read_prediction_txt_file(txt_file):
    colors_list=[]
    with open(txt_file) as file:
        for line in file:
            line=line.split()
            line=line[2].strip('[').strip(']')
            line=line.split(',')
            line=[entry.strip("'") for entry in line]
            colors_list.append(line)
    return colors_list
#####################################################
if __name__ == "__main__":
   #color_accuracy(colors_target,read_prediction_txt_file('dominant_colors.txt'))
   print('done')

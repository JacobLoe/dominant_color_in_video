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
   colors_target=[]
   for child in root[0].iter():
       if child.get('type')=='#ColourRange': #whenever a shot annotation is found, extract the timestamp from the xml
          for child2 in child:
              if child2.tag=='{http://experience.univ-lyon1.fr/advene/ns}content':
                 colors_target.append(child2.text)
   with open('results/results_rgb.txt','w') as file:
        file.write(str(color_accuracy(colors_target,read_prediction_txt_file('predictions/dominant_colors_rgb.txt'))))
   with open('results/results_HSV.txt','w') as file:
        file.write(str(color_accuracy(colors_target,read_prediction_txt_file('predictions/dominant_colors_HSV.txt'))))
   with open('results/results_cie-lab.txt','w') as file:
        file.write(str(color_accuracy(colors_target,read_prediction_txt_file('predictions/dominant_colors_cie-lab.txt'))))
   with open('results/results_reduced_colors_rgb.txt','w') as file:
        file.write(str(color_accuracy(colors_target,read_prediction_txt_file('predictions/dominant_colors_reduced_colors_rgb.txt'))))
   with open('results/results_reduced_colors_HSV.txt','w') as file:
        file.write(str(color_accuracy(colors_target,read_prediction_txt_file('predictions/dominant_colors_reduced_colors_HSV.txt'))))
   with open('results/results_reduced_colors_cie-lab.txt','w') as file:
        file.write(str(color_accuracy(colors_target,read_prediction_txt_file('predictions/dominant_colors_reduced_colors_cie-lab.txt'))))
   with open('results/results_scene_rgb.txt','w') as file:
        file.write(str(color_accuracy(colors_target,read_prediction_txt_file('predictions/dominant_colors_scene.txt'))))
   with open('results/results_scene_HSV.txt','w') as file:
        file.write(str(color_accuracy(colors_target,read_prediction_txt_file('predictions/dominant_colors_scene_HSV.txt'))))
   with open('results/results_scene_cie-lab.txt','w') as file:
        file.write(str(color_accuracy(colors_target,read_prediction_txt_file('predictions/dominant_colors_scene_cie-lab.txt'))))

   print('done')

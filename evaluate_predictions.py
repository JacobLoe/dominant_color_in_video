#####################################################
## libraries
#####################################################
import extract_dominant_color as edc
import zipfile
import xml.etree.ElementTree as ET
#####################################################
## functions
#####################################################
def color_accuracy(colors_target,colors_predictions,*acc_single):
    tl=0
    pl=0
    if acc_single:
        acc=[]
        for target,pred in zip(colors_target,colors_predictions):
            acc.append(len(list(set(target) & set(pred)))/len(target))
        return acc
    else:
        for target,pred in zip(colors_target,colors_predictions):
            if len(pred)<len(target):
                target=target[:len(pred)]
                tl+=len(target)
                pl+=len(list(set(target) & set(pred)))
            else:
                pred=pred[:len(targets)]
                tl+=len(target)
                pl+=len(list(set(target) & set(pred)))
        return pl/tl
#####################################################
def read_prediction_txt_file(txt_file):
    colors_list=[]
    with open(txt_file) as file:
        for line in file:
            line=line.split()
            line_aux=[]
            for i,elem in enumerate(line[2:]):
                if i%2==0:
                    line_aux.append(elem.strip('[').strip(']').strip("'"))
            colors_list.append(line_aux)
    return colors_list
#####################################################
def read_target_colors_azp(azp_path):
    zip_ref = zipfile.ZipFile(azp_path)
    zip_ref.extractall('/tmp')
    tree = ET.parse('/tmp/content.xml')
    root = tree.getroot().findall('./{http://experience.univ-lyon1.fr/advene/ns}annotations')
    colors_target=[]
    for child in root[0].iter():
        if child.get('type')=='#ColourRange':
            for child2 in child:
                if child2.tag=='{http://experience.univ-lyon1.fr/advene/ns}content':
                    colors_target.append(child2.text.split(','))
    return colors_target
#####################################################
if __name__ == "__main__":
   zip_ref = zipfile.ZipFile('/data/scenes/CompanyMen_v1.0-split-012-Bobby_being_angry.azp')
   zip_ref.extractall('/tmp')
   tree = ET.parse('/tmp/content.xml')
   root = tree.getroot().findall('./{http://experience.univ-lyon1.fr/advene/ns}annotations')
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

import fitz
from typing import Tuple
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date
#import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow_hub.keras_layer import KerasLayer
import pathlib
import os
from PIL import Image
import pickle
import shutil
#import sklearn

#import yaml

import easyocr
import numpy as np
from PIL import Image, ImageDraw
import os
import shutil
import re


# conversion en image

def convert_pdf2img(input_file: str, pages: Tuple = None):
    """Converts pdf to image and generates a file by page"""
    # Open the document
    pdfIn = fitz.open(input_file)
    output_files = []
    # Iterate throughout the pages
    for pg in range(pdfIn.pageCount):
        if str(pages) != str(None):
            if str(pg) not in str(pages):
                continue
        # Select a page
        page = pdfIn[pg]
        rotate = int(0)
        # PDF Page is converted into a whole picture 1056*816 and then for each picture a screenshot is taken.
        # zoom = 1.33333333 -----> Image size = 1056*816
        # zoom = 2 ---> 2 * Default Resolution (text is clear, image text is hard to read)    = filesize small / Image size = 1584*1224
        # zoom = 4 ---> 4 * Default Resolution (text is clear, image text is barely readable) = filesize large
        # zoom = 8 ---> 8 * Default Resolution (text is clear, image text is readable) = filesize large
        zoom_x = 2
        zoom_y = 2
        # The zoom factor is equal to 2 in order to make text clear
        # Pre-rotate is to rotate if needed.
        mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pix = page.getPixmap(matrix=mat, alpha=False)
        output_file = str(pathlib.Path('DMR_fraude/depot/TMP/')) +'/'+ f"{os.path.splitext(os.path.basename(input_file))[0]}__page{pg+1}.png"
        pix.writePNG(output_file)
        output_files.append(output_file)
    pdfIn.close()
    summary = {
        "File": input_file, "Pages": str(pages), "Output File(s)": str(output_files)
    }
    # Printing Summary
    print("## Summary ########################################################")
    print("\n".join("{}:{}".format(i, j) for i, j in summary.items()))
    print("###################################################################")
    return output_files



file =[]

#link of folder
directory = pathlib.Path('DMR_fraude/depot/')
for filename in os.listdir(directory):
    if filename.endswith(".pdf"):
        file.append(os.path.join(directory, filename))
        print(os.path.join(directory, filename))

    else:
        continue

# convert image in TMP      
for i in file:
  convert_pdf2img(i)






IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_DEPTH = 3


    

    
def predict_image(Path_class,path, model):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction
    
    Returns
    -------
    Predicted class
    """
    
    names= pd.read_csv(Path_class,names=['Names'])
    #A=np.array(Image.open(path))
    #r, g, b = image_rgb.getpixel((w, h))
    image= [np.array(Image.open(path).convert('RGB').resize((IMAGE_WIDTH, IMAGE_HEIGHT)))]
    #print(image)
    prediction_vector = model.predict(np.array(image))
    #print(prediction_vector)
    predicted_classes = np.argmax(prediction_vector, axis=1)[0]
    names_classes=names['Names'][predicted_classes]
    #print(names)
    #print(predicted_classes)
    return prediction_vector, predicted_classes,names_classes

def load_model(path):
    """Load tf/Keras model for prediction
    """
    return tf.keras.models.load_model(path)


file =[]
directory = pathlib.Path('DMR_fraude/depot/TMP/')# change in function of our folder
file_destination = pathlib.Path('DMR_fraude/check/')# change in function of our folder
file_destination_pixel = pathlib.Path('DMR_fraude/check/modification_pixel')
file_destination_date = pathlib.Path('DMR_fraude/check/date_feriee')
file_destination_ro = pathlib.Path('DMR_fraude/check/non_soumis_ro')
file_destination_ref = pathlib.Path('DMR_fraude/check/ref_arch_faux')
racine= pathlib.Path('DMR_fraude/')
Path_class = pathlib.Path('DMR_fraude/depot/TMP/model/classe_name.txt')
model = load_model(pathlib.Path('C:/Users/nathanael/OneDrive/Documents/MAG3-EBDS/API/model_ame.h5'))
model.summary()




# cree une liste
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        file.append(os.path.join(directory, filename))
        print(os.path.join(directory, filename))

    else:
        continue


reader = easyocr.Reader(['en'])
def detect_text_blocks(img_path):
    detection_result = reader.detect(img_path,
                                 width_ths=0.7, 
                                 mag_ratio=1.5
                                 )
    text_coordinates = detection_result[0][0]
    return text_coordinates

def draw_bounds(img_path, bbox):
    image = Image.open(img_path)  
    draw = ImageDraw.Draw(image)
    for b in bbox:
        p0, p1, p2, p3 = [b[0], b[2]], [b[1], b[2]], \
                         [b[1], b[3]], [b[0], b[3]]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill='red', width=2)
    return np.asarray(image)

# Draw bounding boxes
def draw_boxes(image, bounds, color='yellow', width=2):
    draw = ImageDraw.Draw(image)
    for b in bounds:
        p0, p1, p2, p3 = [b[0], b[2]], [b[1], b[2]], \
                         [b[1], b[3]], [b[0], b[3]]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill='red', width=2)
    return image


def dateferiee(element):
  match=re.findall(r'\d+[/.-]\d+[/.-]\d+', element)
  return match



jour_ferie=["1/1/2022","01-01-2022","01/01/2022",
"18/4/2022","18-04-2022","18/04/2022",
"1/5/2022","01-05-2022","01/05/2022",
"8/5/2022","08-05-2022","08/05/2022",
"26/5/2022","26-05-2022","26/05/2022",
"6/6/2022","06-06-2022","06/06/2022",
"14/7/2022","14-07-2022","14/07/2022",
"15/8/2022","15-08-2022","15/08/2022",
"1/11/2022","01-11-2022","01/11/2022"
"11/11/2022","11-11-2022",
"25/12/2022","25-12-2022",
"1/1/2023","01-01-2023","01/01/2023",
"10/4/2023","10-04-2023","10/04/2023",
"1/5/2023","01-05-2023","01/05/2023",
"8/5/2023","08-05-2023","08/05/2023",
"18/5/2023","18-05-2023","18/05/2023",
"29/5/2023","29-05-2023","29/05/2023",
"14/7/2023","14-07-2023","14/07/2023",
"15/8/2023","15-08-2023","15/08/2023",
"1/11/2023","01-11-2023","01/11/2023",
"11/11/2023","11-11-2023",
"25/12/2023","25-12-2023",
"1/1/2024","01-01-2024","01/01/2024",
"1/4/2024","01-04-2024","01/04/2024",
"1/5/2024","01-05-2024","01/05/2024",
"8/5/2024","08-05-2024","08/05/2024",
"9/5/2024","09-05-2024","09/05/2024",
"20/5/2024","20-05-2024","20/05/2024",
"14/7/2024","14-07-2024","14/07/2024",
"15/8/2024","15-08-2024","15/08/2024",
"1/11/2024","01-11-2024","01/11/2024",
"11/11/2024","11-11-2024",
"25/12/2024","25-12-2024",
"01/01/2025","01-01-2025","1/1/2025",
"21/4/2025","21-04-2025","21/04/2025",
"01/05/2025","01-05-2025","1/5/2025",
"8/5/2025","08-05-2025","08/05/2025",
"29/5/2025","29-05-2025","29/05/2025",
"9/6/2025","09-06-2025","09/06/2025",
"14/7/2025","14-07-2025","14/07/2025",
"15/8/2025","15-08-2025","15/08/2025",
"1/11/2025","01-11-2025","01/11/2025",
"11/11/2025","11-11-2025",
"25/12/2025","25-12-2025",
"1/1/2026","01-01-2026","01/01/2026",
"6/4/2026","06-04-2026","06/04/2026",
"1/5/2026","01-05-2026","01/05/2026",
"8/5/2026","08-05-2026","08/05/2026",
"14/5/2026","14-05-2026","14/05/2026"
"25/5/2026","25-05-2026","25/05/2026",
"14/7/2026","14-07-2026","14/07/2026",
"15/8/2026","15-08-2026","15/08/2026",
"1/11/2026","01-11-2026","01/11/2026",
"11/11/2026","11-11-2026",
"25/12/2026","25-12-2026",
"1/1/2027","01-01-2027","01/01/2027",
"29/3/2027","29-03-2027","29/03/2027",
"1/5/2027","01-05-2027","01/05/2027",
"6/5/2027","06-05-2027","06/05/2027",
"8/5/2027","08-05-2027","08/05/2027",
"17/5/2027","17-05-2027","17/05/2027",
"14/7/2027","14-07-2027","14/07/2027",
"15/8/2027","15-08-2027","15/08/2027",
"1/11/2027","01-11-2027","01/11/2027",
"11/11/2027","11-11-2027",
"25/12/2027","25-12-2027"]





Prestation_sans_ro =["Accompagnement Adulte/Enfant Secteur Conventionné" ,"Accompagnement Ad/Enf SC ","ACCAE",
"Accompagnement Adulte/Enfant Sect non Conventionné","Accompagnement Ad/Enf SNC","ACCAE" ,
"Accompagnement Adulte Secteur Conventionné"       ,"Accompagnement Adulte SC","ACCADU",
"Accompagnement Enfant Cure Thermale"              ,"Accompagnement Enft Cure" ,"ACCCU",
"Accompagnement Enfant Cure Thermale NRSS"          ,"Acc.Enfant Cure NRSS"     ,"ACCCUR",
"Accompagnement Enfant M"                         ,"Accompagnement Enfant M"  ,"ACCM" ,
"Accompagnement Enfant Secteur Conventionné"        ,"Accompagnement Enfant"    ,"ACCG" , 
"Accompagnement Enfant Secteur Conventionné"        ,"Accompagnement Enfant SC" ,"ACC" ,  
"Accompagnement Enfant Secteur non Conventionné"    ,"Accompagnement Enft SNC"  ,"ACC"  , 
"Actes biologie non remboursés Sécurité Sociale"    ,"Actes biologique NR"      ,"B  NR", 
"Actes biologie non remboursés Sécurité Sociale"    ,"Actes biologiques NR"     ,"B NR",  
"Actes Dentaire NR"                                 ,"Actes Dentaire NR"        ,"ACTDNR",
"Acupuncture"                                       ,"Acupuncture"              ,"ACUPON",
"Acupuncture et Homéopathie"                        ,"Acupuncture-Homéopathie"  ,"ACU" ,  
"Aide sociale Adoption"                             ,"Aide sociale Adoption"    ,"AIDADO",
"Aide sociale naissance"                            ,"Aide sociale naissance"  ,"AIDNAI",
"Alèse non remboursée ss"                           ,"Alèse NRSS"               ,"ALESE" ,
"Aliments sans Gluten"                              ,"Aliments sans Gluten"     ,"GLUPA" ,
"Allocation Adoption"                              ,"Allocation Adoption"      ,"ADO" ,  
"Allocation frais de maternité"                     ,"Allocation frais maternit","NAI",   
"Ambulances Agrées Non Remboursables"               ,"Ambulances Agrées NR"     ,"ABANR" ,
"Amniocentèse non Remboursée par Sécurité Sociale"  ,"Amniocentèse non Remb SS" ,"AMNIOC",
"Antipaludéen prescrit NR"                          ,"Antipaludéen prescrit NR" ,"PALU" , 
"Appareillage non Remboursé par la Sécurité Sociale","Appareillage non Remb SS" ,"APPNR", 
"Aromathérapie consult"                             ,"Aromathérapie consult"    ,"AROMAC",
"Aromathérapie produits"                            ,"Aromathérapie produits"   ,"AROMAT",
"Art-thérapie"                                    ,"Art-thérapie"             ,"ARTERA",
"Auriculotthérapie"                                 ,"Auriculotthérapie"        ,"AURIC", 
"Bilan nutritionnel"                                ,"Bilan nutritionnel"       ,"BILNUT",
"Bioénergie et Neurostimulation élec transcutée"    ,"Bioénergie et Neurostimul","BIONEU",
"BONUS DENTAIRE"                                    ,"BONUS DENTAIRE"           ,"BONDEN",
"BONUS OPTIQUE"                                     ,"BONUS OPTIQUE"            ,"OPTBON",
"BONUS OPTIQUE"                                     ,"BONUS OPTIQUE Enfant"     ,"BONOEN",
"BONUS OPTIQUE Faible Corr"                         ,"BONUS OPTIQUE Faibl Corr" ,"BONOPC",
"BONUS OPTIQUE Forte Corr"                          ,"BONUS OPTIQUE Forte Corr" ,"BONOGC",
"BONUS OPTIQUE Lunettes"                           ,"BONUS OPTIQUE Lunettes"   ,"OBONLU",
"BONUS OPTIQUE MUTUELLE"                            ,"BONUS OPTIQUE MUTUELLE"   ,"BONOPT",
"BONUS RENFORT OPTIQUE"                             ,"BONUS RENFORT OPTIQUE"    ,"BONROP",
"Bridges Refusés par la SS"                         ,"Bridges Refusés par la SS","BRIDGR",
"Chambre Maternité SC Supplément"                   ,"Chamb Matern SC Supplémt" ,"CHMATS",
"Chambre Maternité SNC Supplément"                  ,"Chamb Matern SNC Supplémt","CHMATS",
"Chambre Part. avec hébergement SC au délà 8 jours" ,"Chamb.avec hébgt SC +8j"  ,"SHO+8J",
"Chambre Part. avec hébergement SNC au délà 8 jours","Chamb.avec hébgt SNC +8j" ,"SHO+8J",
"Chambre Particulière sans hébergement SNC"      ,"Chamb.sans hébergt SNC"   ,"SHOSH", 
"Chiropractie"                                     ,"Chiropractie"             ,"CHIROP",
"Chirurgie de l'Oeil"                              ,"Chirurgie de l'Oeil"      ,"CHIOPT",
"Chirurgie Oeil droit"                              ,"Chirurgie Oeil droit"     ,"CHIOPD",
"Chirurgie Oeil gauche"                             ,"Chirurgie Oeil gauche"    ,"CHIOPG",
"Coach en développement personnel"                  ,"Coach en develop perso"   ,"COADP", 
"Coach sportif"                                     ,"Coach sportif"            ,"COASPO",
"Compléments alimentaires"                          ,"Compléments alimentaires" ,"COMPAL",
"Contraceptifs prescrits non remboursables"         ,"Contraceptifs presc. NR"  ,"CONPNR",
"Contraception"                                     ,"Contraception"            ,"CONTRA",
"Contraception Féminine"                            ,"Contraception Féminine"   ,"CONFEM",
"Contraception Patch/Pilule/Stérilet"               ,"Contracep Patch Pil Stéri,CONPPS",
"Couche non remboursée ss"                          ,"Couche NRSS"              ,"COUCHE",
"Couronne Dentaire non remboursable sauf provisoire","Couronne Dentaire nonremb","COUNR", 
"Couronne Dentaire non remb. sur dent vivante"      ,"Cour.Dent.NR dent vivante","COUDVI",
"Décès"                                             ,"Décès"                    ,"DCD"  ,
"Dentaire Bonus de fidélité"                        ,"Dentaire Bonus fidélité"  ,"DEBOFI",
"Dent Bonus fid. Orthodontie Acc."                  ,"Dent Bonus fid. Ortho Acc","DEBOOR",
"Dent Bonus fid. Orthodontie Acc. SNC"              ,"Dent Bonus fid. Ortho NC" ,"DEBOOR",
"Dent Bonus fid. Prothèses Acc"                     ,"Dent Bonus fid. Proth Acc","DEBOPR",
"Dent Bonus fid. Prothèses Acc SNC"                 ,"Dent Bonusfid Proth Ac NC","DEBOPR",
"Dep Cancer du Colon"                               ,"Dep Cancer du Colon"      ,"DCACOL",
"Dep Infection urinaire"                            ,"Dep Infection urinaire"   ,"DINFUR",
"Depistage du cancer du col de l'uterus"            ,"Dep Cancer Col Uterus"    ,"DCCU", 
"Dépistage du Cancer du Sein"                       ,"Dépistage Cancer du Sein" ,"DCS",  
"Dépistage H1N1 ou VIH"                             ,"Dépistage H1N1 ou VIH"    ,"H1NVIH",
"Dépistage Hépatite B"                              ,"Dep Hépatite B"           ,"DHEPAB",
"Dep Polyarthrite rhumotoid"                        ,"Dep Polyart rhumotoid"    ,"DPORHU",
"Diététiciens"                                      ,"Diététiciens"             ,"DIET",  
"Endodontie non remboursée "                        ,"Endodontie non remboursée","ENDO",  
"Ergothérapie"                                      ,"Ergothérapie"            ,"ERGOTE",
"Etiopathie"                                        ,"Etiopathie"               ,"ETIOPA",
"Fécondation in Viro Non remboursée"                ,"Fécondation in Viro NR"   ,"FIVNR",
"Forfait Journalier"                                ,"Forfait Journalier"       ,"FJ",    
"Forfait Journalier Jour de Sortie"                 ,"Forfait Journalier Sortie","FJA",  
"Forfait Journalier Secteur non conventionné"       ,"Forfait Journalier SNC"   ,"FJ",    
"Forfait Journalier Sortie Secteur non Conventionné","FJ Sortie SNC"            ,"FJA",  
"Forfait Mariage"                                   ,"Forfait Mariage"          ,"MARIAG",
"Forfait Prothèses Fixes Hors Nomenclature"         ,"Forfait Proth Fixes HN"   ,"SPRFFH",
"Forfait soins divers NR"                           ,"Forfait soins divers NR"  ,"FSNR", 
"Frais de Téléphone sur justificatifs"              ,"Frais Téléphone"          ,"TEL",   
"Frais de Télévision"                               ,"Frais de Télévision"      ,"TV",    
"Frais de Télévision en Chirurgie"                  ,"Frais de Télévision Chir" ,"TVCH", 
"Frais de Télévision en Medecine"                   ,"Frais de Télévision Med"  ,"TVME",  
"Frais de Télévision Hospitalisation"               ,"Frais de Télévision Hosp.","TVHOSP",
"Frais de Télévision Hospitalisation SNC"           ,"Frais de Télévision H SNC","TVHOSP",
"Frais Funéraire"                                   ,"Frais Funéraire"          ,"FRAFUN",
"Frais Hospitalisation TV et Téléphone chirurgie"   ,"Frais Hosp Tv Tel Chir"   ,"TVTEL1",
"Frais Hospitalisation TV et Téléphone maternité"   ,"Frais Hosp Tv Tel Mat"    ,"TVTEL7",
"Frais Hospitalisation TV et Téléphone médicale"    ,"Frais Hosp Tv Tel Médi"   ,"TVTEL2",
"Frais Hospitalisation TV et Téléphone repos conv"  ,"Frais Hosp Tv Tel Repos"  ,"TVTEL6",
"Frais journaux Hospitalisation sous Justificatifs" ,"Frais Journaux"          ,"JOURNO",
"Gemmothérapie produits"                            ,"Gemmothérapie produits"   ,"GEMMOT",
"Homéopathie consultation NR"                       ,"Homéopathie consult NR"   ,"HOMEOC",
"Homéopathie Prescirte non remboursée"              ,"Homéopathie prescrite NR" ,"HOMPNR",
"Huiles Essentielles"                               ,"Huiles Essentielles"      ,"HUILES",
"Huiles et gélules CBD"                             ,"Huiles et gélules CBD"    ,"CBD",  
"Huiles et gélules CBD non remboursé"               ,"Huiles et gélules CBD NR" ,"CBD-NR",
"Huiles et produits DETOX"                          ,"Huiles et produits DETOX" ,"DETOX", 
"Hypnose"                                           ,"Hypnose"                  ,"HYPNOS",
"Implant"                                           ,"Implant"                  ,"IMPLAN",
"Implantologie non remboursée"                      ,"Implantologie non remb."  ,"IMPR",  
"Inlay et Onlay non remboursé"                      ,"Inlay Onlay non remboursé","INLAYR",
"IVG Non remboursée"                                ,"IVG NR"                   ,"IVGNR", 
"Kiné Méthode Mézières NR"                          ,"Kiné Méthode Mézières NR","MEZIER",
"Kinésiologie"                                      ,"Kinésiologie"             ,"KISIO", 
"Licence sportive"                                  ,"Licence sportive"         ,"LISPOR",
"Médecines Douces"                                  ,"Médecines Douces"         ,"MEDDOU",
"Médecines Nouvelles"                               ,"Médecines Nouvelles"      ,"MEDNOU",
"Médicaments non Remboursable"                      ,"Médicaments non Remboursé","MEDNR", 
"Médicaments prescrits non Remboursables"           ,"Méd prescrits non Remb"   ,"MEDPNR",
"Mésothérapie"                                     ,"Mésothérapie"             ,"MESOTH",
"Micro Kiné"                                       ,"Micro Kiné"               ,"MICROK",
"Monture Optique Forfait1A Contrat Resp"            ,"Monture Optique F1A"      ,"LUNF1A",
"Monture Optique Forfait1 Contrat Resp"             ,"Monture Optique F1"       ,"LUNF1", 
"Monture Optique Forfait2A Contrat Resp"            ,"Monture Optique F2A"      ,"LUNF2A",
"Monture Optique Forfait2 Contrat Resp"             ,"Monture Optique F2"       ,"LUNF2",
"Monture Optique Forfait3A Contrat Resp"            ,"Monture Optique F3A"      ,"LUNF3A",
"Monture Optique Forfait3 Contrat Resp"             ,"Monture Optique F3"       ,"LUNF3", 
"Musicothérapie"                                    ,"Musicothérapie"           ,"MUSIT", 
"Naturopathie"                                      ,"Naturopathie"             ,"NATURO",
"Nutritionniste"                                   ,"Nutritionniste"           ,"NUTRI", 
"Olfactothérapie"                                   ,"Olfactothérapie"          ,"OLFAC", 
"Oligothérapie"                                     ,"Oligothérapie"            ,"OLIGO", 
"Optique Bonus de fidélité"                         ,"Optique Bonus de fidélité","OPBOFI",
"Orthodontie Adulte Non remboursée sécu +"          ,"Orthodontie Adulte NR+"   ,"ORTNR+",
"Orthodontie hors nomenclature"                    ,"Orthodontie HN"           ,"ORTHN", 
"Ostéodensitométrie"                                ,"Ostéodensitométrie"       ,"OSTDEN",
"Ostéopathie"                                       ,"Ostéopathie"              ,"OST",   
"Ostéoporose analyses"                              ,"Ostéoporose"              ,"OSTPO",
"Parodontologie Non Remboursée"                     ,"Parodontologie Non Remb"  ,"TDSR",  
"Parodontologie refusée"                            ,"Parodontologie refusée"   ,"PAR",   
"Participation Mut Pile entreti Prothè­se Auditive" ,"Part/Pile Entr P.Auditive","PILENT",
"Participation Mutuelle 1er Verre Optique Multfocal","Part.Mut 1er Verre multif","VER1MF",
"Participation Mutuelle 1er Verre Optique Unifocal" ,"Part.Mut 1er Verre unifoc","VER1UF",
"Participation Mutuelle / 1er Verre Opt progressif" ,"Part.Mut/1er Verre Prog"  ,"VERP1" ,
"Participation Mutuelle 2ème Verre Optique Unifocal","Part.Mut 2e Verre unifoc" ,"VER2UF",
"Participation Mutuelle / 2ème Verre Opt progressif","Part.Mut/2ème Verre Prog" ,"VERP2", 
"Participation Mutuelle 2me Verre Optique Multfocal","Part.Mut 2e Verre multfoc","VER2MF",
"Participation Mutuelle Achat Prothè­se Auditive"   ,"Part./Achat Prot.Auditive","PAUM",  
"Participation Mutuelle Cure Refusée SS"            ,"Part.Mut.Cure Refusée SS" ,"CURR",  
"Participation Mutuelle Lentilles non RembSS Enfant","Part.Mut/Lentill NRSS Enf","LENMRE",
"Participation Mutuelle sur 1ère Lentille NRSS SMIB","1e Lentille NRSS SMIB"    ,"LEN1R", 
"Participation Mutuelle sur 1­re Lentille NRSS"     ,"Part.Mut.1e Lentille NRSS","LENM1R",
"Participation Mutuelle sur 2ème Lentille NRSS SMIB","2e Lentille NRSS SMIB"   ,"LEN2R", 
"Participation Mutuelle sur 2­me Lentille NRSS"     ,"Part.Mut.2e Lentille NRSS","LENM2R",
"Participation Mutuelle sur Lentilles non Remb.SS"  ,"Part.Mut./Lentilles NRSS" ,"LENMR", 
"Participation Mutuelle sur Lunettes non Remb SS"   ,"Part.Mut / Lunettes NRSS" ,"OPTMR", 
"Participation Mutuelle sur Orthodontie NR"         ,"Part.Mut/Orthodontie NR"  ,"PORTNR",
"Participation Mutuelle / Transport non remboursé"  ,"Part./Transport non Remb.","PTPNR",
"Part.Mut/Prothèses Dentaires non remboursées"      ,"Part.Mut/Prothèses DentNR","PPRONR",
"Part Mutuelle 1er Verre Optique Multifocal C BLEU" ,"Part.Mut 1 Verre MF CBleu","V1MFCB",
"Part Mutuelle 1er Verre Optique Unifocal C BLEU"   ,"Part.Mut 1 Verre UF CBleu","V1UFCB",
"Part Mutuelle 2me Verre Optique Multifocal C BLEU" ,"Part.Mut 2 Verre MF CBleu","V2MFCB",
"Part Mutuelle 2me Verre Optique Unifocal C BLEU"   ,"Part.Mut 2 Verre UF CBleu","V2UFCB",
"Part Mutuelle Lentilles non RembSS Faible correct" ,"Part.Mut/Lent NRSS FaibCo","LENMRP",
"Part Mutuelle Lentilles non RembSS Forte correct"  ,"Part.Mut/Lent NRSS FortCo","LENMRG",
"Part.Mutuelle/Lentilles non Remb.SS Luxembourg"    ,"Part.Mut/Lentilles NRSS L","LLENMR",
"Part Mutuelle Prothèse Orthopédique non remboursée","Part/Prot.Orthopédique NR","PPORNR",
"Part Mutuelle Verre droit faible correction"       ,"PartMut Verre D Simple PC","VERDSP",
"Part Mutuelle Verre droit forte correction"       ,"PartMut Verre D Simple GC","VERDSG",
"Part Mutuelle Verre droit Multifocal Sphérique"    ,"PartMut VerreD M-Sphériqu","VERDMS",
"Part Mutuelle Verre droit Multif.Sphérocylindrique","PartMut VerreD M-Sphé.Cyl","VERDMC",
"Part Mutuelle Verre droit Optique FaibleCorrection","PartMut Verre D Faibl.Cor","VERDPC",
"Part Mutuelle Verre droit Optique Forte Correction","PartMut Verre D Forte Cor","VERDGC",
"Part Mutuelle Verre droit Optique Simple"          ,"PartMut Verre D Simple"   ,"VERDS", 
"Part Mutuelle Verre droit Unif. Faible Correction" ,"PartMut VerreD Uni-FaibCo","VERDUP",
"Part Mutuelle Verre droit Unif. Forte Correction"  ,"PartMut VerreD Uni-FortCo","VERDUG",
"Part Mutuelle Verre droit Unifocal Sphérique"      ,"PartMut VerreD U-Sphériqu","VERDUS",
"Part Mutuelle Verre droit Unif. Sphérocylindrique" ,"PartMut VerreD U-Sphé.Cyl","VERDUC",
"Part Mutuelle Verre gauche faible correction"     ,"PartMut Verre G Simple PC","VERGSP",
"Part Mutuelle Verre gauche forte correction"       ,"PartMut Verre G Simple GC","VERGSG",
"Part Mutuelle Verre gauche Multifocal Sphérique"   ,"PartMut VerreG M-Sphériqu","VERGMS",
"Part Mutuelle Verre gaucheMultif.Sphérocylindrique","PartMut VerreG M-Sphé.Cyl","VERGMC",
"Part Mutuelle Verre gauche Optique FaiblCorrection","PartMut Verre G Faibl.Cor","VERGPC",
"Part Mutuelle Verre gauche Optique ForteCorrection","PartMut Verre G Forte Cor","VERGGC",
"Part Mutuelle Verre gauche Optique Simple"         ,"PartMut Verre G Simple"   ,"VERGS", 
"Part Mutuelle Verre gauche Unif. Faible Correction","PartMut VerreG Uni-FaibCo","VERGUP",
"Part Mutuelle Verre gauche Unif. Forte Correction" ,"PartMut VerreG Uni-FortCo","VERGUG",
"Part Mutuelle Verre gauche Unifocal Sphérique"     ,"PartMut VerreG U-Sphériqu","VERGUS",
"Part Mutuelle Verre gauche Unif. Sphérocylindrique","PartMut VerreG U-Sphé.Cyl","VERGUC",
"Part Mutuelle Verres Optiques Faible Correction"   ,"Part.Mut/Verres Faibl Cor","VERMPC",
"Part Mutuelle Verres Optiques Forte Correction"    ,"Part.Mut/Verres Forte Cor","VERMGC",
"Part Mut Verre D Optique Cor mixte AF Cont.Res"    ,"PartMut Ver D mixte AF CR","VDCRAF",
"Part Mut Verre D Optique Cor mixte AF PNR"         ,"PartMut VerD mixte AF PNR"",VDNRAF",
"Part Mut Verre D Optique Cor mixte CF Cont.Res"    ,"PartMut Ver D mixte CF CR","VDCRCF",
"Part Mut Verre D Optique Cor mixte CF PNR"         ,"PartMut VerD mixte CF PNR","VDNRCF",
"Part Mut Verre D Optique Correction mixte Cont.Res","PartMut Ver D Co mixte CR","VDCRAC",
"Part Mut Verre D Optique Correction mixte PNR"     ,"PartMut VerD Co mixte PNR","VDNRAC",
"Part Mut Verre droite Optique Multifocal"         ,"PartMut Verre D Multifoc.","VERDMU",
"Part Mut Verre droit Optique Cor Mixte ACS"        ,"PartMut VerD Cor Mixt ACS","VDCSAC",
"Part Mut Verre droit Optique Faibl.Cor ACS"        ,"PartMut VerD Faib.Cor ACS","VDCSA",
"Part Mut Verre droit Optique Faibl.Cor Contrat Res","PartMut Ver D Faib.Cor CR","VDCRA", 
"PartMut Verre droit Optique Faibl.Cor PNR"         ,"PartMut VerD Faib.Cor PNR","VDNRA",
"Part Mut Verre droit Optique Forte.Cor ACS"     ,"PartMut VerD Fort Cor ACS","VDCSC", 
"Part Mut Verre droit Optique Forte Cor Contrat Res","PartMut Ver D Fort.Cor CR","VDCRC",
"Part Mut Verre droit Optique Forte Cor PNR"        ,"PartMut VerD Fort.Cor PNR","VDNRC", 
"Part Mut Verre droit Optique Hyp.compl Contrat Res","PartMut Ver D Hyp comp CR","VDCRF", 
"Part Mut Verre droit Optique Hyp.compl PNR"        ,"PartMut VerD Hyp comp PNR","VDNRF", 
"Part Mut Verre droit Optique Multifocal/Progressif","PartMut Verre D MulF/Prog","VERDMP",
"Part Mut Verre droit Optique Unifocal"             ,"PartMut Verre D Unifocal" ,"VERDU", 
"Part Mut Verre droit Progressif faible correction" ,"PartMut Verre D Prog PC"  ,"VERDPP",
"Part Mut Verre droit Progressif forte correction"  ,"PartMut Verre D Prog GC"  ,"VERDPG",
"Part Mut Verre gauche Optique Faibl.Cor ACS"       ,"PartMut VerG Faib.Cor ACS","VGCSA", 
"PartMut Verre gauche Optique Faibl.Cor PNR"        ,"PartMut VerG Faib.Cor PNR","VGNRA", 
"Part Mut Verre gauche Optique Forte.Cor ACS"       ,"PartMut VerG Fort Cor ACS","VGCSC",
"Part Mut Verre gauche Optique Forte Cor PNR"       ,"PartMut VerG Fort.Cor PNR","VGNRC", 
"Part Mut Verre gauche Optique Hyp.compl PNR"       ,"PartMut VerG Hyp comp PNR","VGNRF", 
"Part Mut Verre gauche Optique Multifocal"          ,"PartMut Verre G Multifoc.","VERGMU",
"Part Mut Verre Gauche Optique Unifocal"            ,"PartMut Verre G Unifocal" ,"VERGU", 
"Part Mut Verre gauche Progressif faible correction","PartMut Verre G Prog PC"  ,"VERGPP",
"Part Mut Verre gauche Progressif forte correction" ,"PartMut Verre G Prog GC"  ,"VERGPG",
"Part Mut Verre gauch Optique FaibleCor Contrat Res","PartMut Ver G Faib.Cor CR","VGCRA", 
"Part Mut Verre gauch Optique Forte Cor Contrat Res","PartMut Ver G Fort.Cor CR","VGCRC", 
"Part Mut Verre gauch Optique Hyp compl Contrat Res","PartMut Ver G Hyp.comp CR","VGCRF", 
"Part Mut Verre gauch Optique Multifocal/Progressif","PartMut Verre G MulF/Prog","VERGMP",
"Part Mut Verre G Optique Cor mixte AF Cont.Res"    ,"PartMut Ver G mixte AF CR","VGCRAF",
"Part Mut Verre G Optique Cor mixte AF PNR"         ,"PartMut VerG mixte AF PNR","VGNRAF",
"Part Mut Verre G Optique Cor mixte CF Cont.Res"    ,"PartMut Ver G mixte CF CR","VGCRCF",
"Part Mut Verre G Optique Cor mixte CF PNR"         ,"PartMut VerG mixte CF PNR","VGNRCF",
"Part Mut Verre G Optique Correction mixte Cont.Res","PartMut Ver G Co mixte CR","VGCRAC",
"Part Mut Verre G Optique Correction mixte PNR"     ,"PartMut VerG Co mixte PNR","VGNRAC",
"Patch Antidouleur"                                 ,"Patch Antidouleur"       ,"PATDOU",
"Patch Anti Tabac"                                  ,"Patch Anti Tabac"         ,"PATTAB",
"Patch Contraceptif"                                ,"Patch Contraceptif"       ,"PATCON",
"Patch Dentaire Blanchissant non remboursé"         ,"Patch Dent BlanchissantNR","PATBLA",
"Patch Minceur"                                     ,"Patch Minceur"            ,"PATMIN",
"Pédicure (Médecine Douce Plus)"                    ,"Pédicure"                ,"PEDI",  
"Pédicure non remboursée ss"                        ,"Pédicure NRSS"            ,"PEDINR",
"Péridurale non Remboursée par la Sécurité Sociale" ,"Péridurale non Remb SS"   ,"PERIDU",
"Pharmacie non Remboursable"                        ,"Pharma. non Remboursable" ,"PHNR",  
"Pharmacie non Remboursable"                        ,"Pharm non Remboursable"   ,"PH NR", 
"Pharmacie non Remboursable"                        ,"Pharm.non Remboursable"   ,"PHN",   
"Pharmacie prescrite non Remboursable"              ,"Pharm prescrite non Remb" ,"PHPNR", 
"Physiothérapie"                                    ,"Physiothérapie"           ,"PHYSIO",
"Phytothérapie "                                    ,"Phytothérapie"            ,"PHYTO", 
"Phytothérapie consultation NR"                     ,"Phytothérapie consult NR" ,"PHYTOC",
"Pilules Contraceptives"                            ,"Pilules Contraceptives"   ,"PILCON",
"Plaque Métal Non remboursée par Sécurité Sociale"  ,"Plaque Métal NR SS"       ,"PL MET",
"Podologie non remboursée ss"                       ,"Podologie NRSS"           ,"PODONR",
"Podologie non remboursée ss complément"            ,"Podologie NRSS complément","PODCNR",
"Posthectomie non remboursée"                       ,"Posthectomie NR"          ,"CIRCON",
"Prélèvement Sanguin non Remboursé par SS"          ,"Prélèvement Sanguin NR"   ,"PB NR", 
"Prélèvement Sanguin non Remboursé par SS"          ,"Prél Sanguin NR"          ,"TB NR", 
"Préparations magistra. NR"                         ,"Préparations magistra. NR","PREPNR",
"Préservatifs"                                      ,"Préservatifs"             ,"PRESER",
"Prime de Mariage"                                  ,"Prime de Mariage"         ,"MARIA", 
"Probiotiques"                                      ,"Probiotiques"             ,"PROBIO",
"Prot dentaire hors nomenclature"                   ,"Prot hors nomenclature"   ,"PROHN",
"Prothèse amovible définitive métalique non remb."  ,"Prot. amov. déf. métal NR","PAMR", 
"Prothèse amovible  définitive résine non remb."    ,"Prot. amov déf. résine NR","PARR", 
"Prothèse Dentaire par Chirurgien Dentiste HN"      ,"Prothèse Dentaire HN"     ,"SPRHN", 
"Prothèse dentaire provisoire non remboursée"       ,"Proth. dent provisoire NR","PDTR",  
"Prothèse fixe céramique non remboursée"            ,"Proth. fixe céramique NR" ,"PFCR",  
"Prothèse fixe Métallique non remboursée"           ,"Proth. fixe Métalique NR" ,"PFMR",  
"Psychologie"                                       ,"Psychologie"              ,"PSYCHO",
"Psychologue entreteien d'évaluation"               ,"Entretien Evaluation Psy" ,"EEP",   
"Psychomotricité"                                   ,"Psychomotricité"         ,"PSYMOT",
"Psychothérapie"                                    ,"Psychothérapie"           ,"PSYTHE",
"Radiologie NR"                                     ,"Radiologie NR"            ,"RADINR",
"Radiologie prescrite NR"                           ,"Radiologie prescrite NR"  ,"RADPNR",
"Radio Scan Dentaire NR"                            ,"Radio Scan Dentaire NR"   ,"RSCDNR",
"Rapatriement de Corps"                             ,"Rapatriement de Corps"    ,"R  COR",
"Réflexologie"                                      ,"Réflexologie"             ,"REFLEX",
"Réflexologie plantaire"                            ,"Réflexologie plantaire"   ,"REFPLA",
"Réparation sur prothèse dentaire non remboursée"   ,"Réparation prot. dent. NR","RPNR",  
"séance accompagnement psychologue"                 ,"Accompagnement Psy"       ,"APS",  
"Séance Psychologue COVID"                          ,"Séance Psychologue COVID" ,"PSYCOV",
"Sexologie"                                         ,"Sexologie"                ,"SEXOLO",
"Shiatsu"                                           ,"Shiatsu"                  ,"SHIAT", 
"Soins Conservateurs par Chirurgien Dentiste HN"    ,"Soins Conserv/Dentiste HN","SC HN", 
"Sophrologie"                                       ,"Sophrologie"             ,"SOPHRO",
"Substitut nicotinique"                             ,"Substitut nicotinique"    ,"TNS",   
"Supplément Chambre Particulière avec hébergement"  ,"Chamb.avec hébergement"   ,"SHOG",  
"Supplément de Chambre Particulière G SNC"        ,"Chamb.Particulière G SNC","SHOG",  
"Supplément de Chambre Particulière SNC"        ,"Chamb.Particulière SNC"  ,"SHO",   
"Supplémnt Chambre Particulière avec hébergement SC","Chamb.avec hébergement SC","SHO",  
"Supplémnt Chambre Particulière sans hébergemen sc" ,"Chamb.sans hébergement sc","SHOSHG",
"Supplémnt Chambre Particulière sans hébergement SC","Chamb.sans hébergement SC","SHOSH", 
"Test de Grossesse"                                 ,"Test de Grossesse"        ,"TESTGR",
"Test Grossesse"                                    ,"Test Grossesse"          ,"TEST",  
"Thérapie de couple"                                ,"Thérapie de couple"       ,"THECOU",
"Traitement Anti Tabagique"                         ,"Traitement Anti Tabagique","TABAC", 
"Traitement Anti Tabagique prescit"                 ,"Traitement AntiTabac Pr." ,"TABACP",
"Traitement de l'Obésité"                           ,"Traitement de l'Obésité"  ,"OBES",
"Traitement des Addictions"                         ,"Traitement des Addictions","ADDIC",
"Traitement des Allergies"                          ,"Traitement des Allergies" ,"ALLER", 
"Traitement Laser Myopie"                           ,"Traitement Laser Myopie"  ,"LASMYO",
"Traitement Laser Myopie Enfant"                    ,"Trait Laser Myopie Enfant","MYOENF",
"Traitement Laser Myopie Faible Correction"         ,"Trait Laser Myopie FaibCo","MYOPC", 
"Traitement Laser Myopie Forte Correction"         ,"Trait Laser Myopie FortCo","MYOGC", 
"Traitement Laser Myopie Oeil Droit"               ,"Trait Laser Myopie Oeil 1","LASMY1",
"Traitement Laser Myopie Oeil Gauche"               ,"Trait Laser Myopie Oeil 2","LASMY2",
"Transport prescrit non remboursé hors cure & hosp.","Transport non remboursé"  ,"TRANR", 
"Vaccin Antigrippe"                                 ,"Vaccin Antigrippe"        ,"ANTGR" 
"Vaccin non Remboursable par la Sécurité Sociale"   ,"Vaccin NR par SS"         ,"VAC",   
"Vaccin prescrit non remboursable SS"              ,"Vaccin prescrit non NR"   ,"VACPNR",
"Vaccin prescrit ou non et non remboursable SS"     ,"Vaccin prescrit ou non NR","VACNPR",
"VACCINS REFUSES PAR SS"                            ,"VACCINS REFUSES PAR SS"   ,"VACR",  
"Verres Optiques Forfait1A Cont. Resp"              ,"Verres Optiques F1A"      ,"VERF1A",
"Verres Optiques Forfait1 Cont. Resp"              ,"Verres Optiques F1"       ,"VERF1", 
"Verres Optiques Forfait2A Cont. Resp"              ,"Verres Optiques F2A"      ,"VERF2A",
"Verres Optiques Forfait2 Cont. Resp"               ,"Verres Optiques F2"      ,"VERF2", 
"Verres Optiques Forfait3A Cont. Resp"              ,"Verres Optiques F3A"      ,"VERF3A",
"Verres Optiques Forfait3 Cont. Resp"               ,"Verres Optiques F3 "      ,"VERF3" ]

for i in file:
  text_coordinates = detect_text_blocks(i)
  recognition_results = reader.recognize(i,
                                 horizontal_list=text_coordinates,
                                 free_list=[]
                                 )
  tr=[]
  for txt in recognition_results:
    tr.append((txt[1]))
  liste= " ".join(tr)
  if len(re.findall('ref',liste))==1:
        res = liste.split('ref', maxsplit=2)[-1]\
                .split(maxsplit=1)[0:2]
        quant = res[1].split(" ")[0]
        date_b= re.findall(r'\d+[/.-]\d+[/.-]\d+', liste)
        d0 = date(int(date_b[-1].split('/')[-1]),1, 1)
        d1 = date(int(date_b[-1].split('/')[-1]),int(date_b[-1].split('/')[1]), int(date_b[-1].split('/')[0]))
        delta = d1 - d0 
        jour= delta.days 
        if (len(str(jour))== 1):
                d= str(date_b[-1].split('/')[-1])
                d= d[2:]+"00"+ str(jour)
                name= os.path.basename(i)
                name_b =str(name).split("__")[0]
                if (quant[:5] == d ):

                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", racine)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", racine)
                    except:
                        print("probleme de redirection des factures frauduleuses 1")
                        print(i)
                        print(quant[:5])
                        print(d)
                else :
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", file_destination_ref)
                        print(i)
                        print(quant[:5])
                        print(d)
                    except:
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".xml", file_destination_ref)
                        print("probleme de redirection des factures non frauduleuses 1")
                        print(i)
                        print(quant[:5])
                        print(d)
        elif (len(str(jour))== 2):
                e= str(date_b[-1].split('/')[-1])
                e= e[2:]+"0"+str(jour)
                name= os.path.basename(i)
                name_b =str(name).split("__")[0]
                if (quant[:5] == e ):
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", racine)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", racine)
                    except:
                        print("probleme de redirection des factures frauduleuses 2")
                        print(i)
                        print(quant[:5])
                        print(e)
                else :
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", file_destination_ref)
                        print(i)
                        print(quant[:5])
                        print(e)
                    except:
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".xml", file_destination_ref)
                        print("probleme de redirection des factures non frauduleuses 2")
                        print(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf")
                        print(i)
                        print(quant[:5])
                        print(e)
                        
        else:
                cday= str(date_b[-1].split('/')[-1])
                cday = cday[2:]+ str(jour)
                name= os.path.basename(i)
                name_b =str(name).split("__")[0]
                if (quant[:5] == cday):
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", racine)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", racine)
                    except:
                        print("probleme de redirection des factures frauduleuses 3")
                        print(i)
                        print(quant[:5])
                        print(cday)
                        
                else :
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", file_destination_ref)
                        print(i)
                        print(quant[:5])
                        print(cday)
                    except:
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".xml", file_destination_ref)
                        print("probleme de redirection des factures non frauduleuses 3")
                        print(i)
                        print(quant[:5])
                        print(cday)
  elif len(re.findall('ref',liste)) > 1:
    res = liste.split('ref', maxsplit=1)[-1]\
                .split(maxsplit=1)[0:2]
    quant = res[1].split(" ")[0]
    date_b= re.findall(r'\d+[/.-]\d+[/.-]\d+', liste)
    d0 = date(int(date_b[1].split('/')[-1]),1, 1)
    d1 = date(int(date_b[1].split('/')[-1]),int(date_b[1].split('/')[1]), int(date_b[1].split('/')[0]))
    delta = d1 - d0 
    jour= delta.days 
    if (len(str(jour))== 1):
                d= str(date_b[1].split('/')[-1])
                d=d[2:]+"00"+ str(jour)
                name= os.path.basename(i)
                name_b =str(name).split("__")[0]
                if (quant[:5] == d ):
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", racine)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", racine)
                    except:
                        print("probleme de redirection des factures frauduleuses 4")
                        print(i)
                        print(quant[:5])
                        print(d)
                else :
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", file_destination_ref)
                        print(i)
                        print(quant[:5])
                        print(d)
                    except:
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".xml", file_destination_ref)
                        print("probleme de redirection des factures non frauduleuses 4")
                        print(i)
                        print(quant[:5])
                        print(d)
    elif (len(str(jour))== 2):
                e= str(date_b[1].split('/')[-1])
                e=e[2:]+"0"+str(jour)
                name= os.path.basename(i)
                name_b =str(name).split("__")[0]
                if (quant[:5] == e ):
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", racine)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", racine)
                    except:
                        print("probleme de redirection des factures frauduleuses 5")
                        print(e)
                        print(i)
                        print(quant[:5])
                else :
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", file_destination_ref)
                    except:
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".xml", file_destination_ref)
                        print("probleme de redirection des factures non frauduleuses 5")
                        print(e)
                        print(i)
                        print(quant[:5])
    else:
                cday= str(date_b[1].split('/')[-1])
                cday= cday[2:]+ str(jour)
                name= os.path.basename(i)
                name_b =str(name).split("__")[0]
                if (quant[:5] == cday):
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", racine)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", racine)
                    except:
                        print("probleme de redirection des factures frauduleuses 6")
                else :
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf",file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", file_destination_ref)
                    except:
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".xml", file_destination_ref)
                        print("probleme de redirection des factures non frauduleuses 6")
                        print(cday)
                        print(quant[:5])
    j = 1
    while j < len(re.findall('ref',liste)):

        res = res[1].split('ref', maxsplit=1)[-1]\
                .split(maxsplit=1)[0:2]
        quant = res[1].split(" ")[0]
        date_b= re.findall(r'\d+[/.-]\d+[/.-]\d+', liste)
        d0 = date(int(date_b[j+1].split('/')[-1]),1, 1)
        d1 = date(int(date_b[j+1].split('/')[-1]),int(date_b[1].split('/')[1]), int(date_b[1].split('/')[0]))
        delta = d1 - d0 
        jour= delta.days 
        if (len(str(jour))== 1):
                d= str(date_b[j+1].split('/')[-1])
                d=d[2:]+"00"+ str(jour)
                name= os.path.basename(i)
                name_b =str(name).split("__")[0]
                if (quant[:5] == d ):
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", racine)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", racine)
                    except:
                        print("probleme de redirection des factures frauduleuses 7")
                        print(d)
                        print(quant[:5])
                else :
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", file_destination_ref)
                    except:
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".xml", file_destination_ref)
                        print("probleme de redirection des factures non frauduleuses 7")
                        print(d)
                        print(quant[:5])
        elif (len(str(jour))== 2):
                e= str(date_b[j+1].split('/')[-1])
                e=e[2:]+"0"+str(jour)
                name= os.path.basename(i)
                name_b =str(name).split("__")[0]
                if (quant[:5] == e ):
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", racine)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", racine)
                    except:
                        print("probleme de redirection des factures frauduleuses 8")
                else :
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", file_destination_ref)
                    except:
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".xml", file_destination_ref)
                        print("probleme de redirection des factures non frauduleuses 8")
                        print(e)
                        print(quant[:5])
        else:
                cday= str(date_b[j+1].split('/')[-1])
                cday =cday[2:]+ str(jour)
                name= os.path.basename(i)
                name_b =str(name).split("__")[0]
                if (quant[:5] == cday):
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", racine)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", racine)
                    except:
                        print("probleme de redirection des factures frauduleuses 9")
                else :
                    try :
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", file_destination_ref)
                    except:
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".pdf", file_destination_ref)
                        shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".xml", file_destination_ref)
                        print("probleme de redirection des factures non frauduleuses 9")
                        print(cday)
                        print(quant[:5])
        
        j += 1
      
    

  else:
        
        name= os.path.basename(i)
        name_b =str(name).split("__")[0]

        try :
            shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", racine)
            shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", racine)
        except:
            print("probleme de redirection des factures non frauduleuses 10")
            print(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf")











for i in file:
    text_coordinates = detect_text_blocks(i)
    recognition_results = reader.recognize(i,
                                 horizontal_list=text_coordinates,
                                 free_list=[]
                                 )
    tr=[]
    for txt in recognition_results:
        tr.append((txt[1]))
    liste= "".join(tr)
    try:
        for j in dateferiee(liste):
            if j in jour_ferie:
                name= os.path.basename(i)
                name_b =str(name).split("__")[0]
                print(j)
                try :
                    shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".pdf", file_destination_date)
                    shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".xml", file_destination_date)
                except:
                    print("plus de factures")
            else:
                print("pas de date feriée")
    except:
        print("pas de date")





#def appartient(elt, lst):
    #for e in lst:
        #if ((e in elt) & (e  in Prestation_sans_ro)) :
            #print("soupçon de fraude")
        #else:
            #print("Absence de fraude")



def appartient(elt, lst):
    for e in lst:
        if e in elt:
            for i in lst:
                if i in Prestation_sans_ro:
                    return "soupçon de fraude"
                else:
                    return " Absence de fraude"
        else:
            return "Absence de fraude"






#tr = []
#for txt in recognition_results:
    #tr.append((txt[1]))

    
for i in file:
    text_coordinates = detect_text_blocks(i)
    recognition_results = reader.recognize(i,
                                 horizontal_list=text_coordinates,
                                 free_list=[]
                                 )
    tr=[]
    for txt in recognition_results:
        tr.append((txt[1]))
    if appartient(["Part Ro","Ro","Régime obligatoire", "PART RO","REGIME OBLIGATOIRE","part ro","ro","RO"], tr) == "soupçon de fraude":
        name= os.path.basename(i)
        name_b =str(name).split("__")[0]
        try :
            shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".pdf", file_destination_ro)
            shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".xml", file_destination_ro)
        except:
            print("plus de factures")
    else:
        print("ce n'est pas une facture avec un regime obligatoire et une prestation non soumise au regime obligatoire")



        #os.chdir('DMR_fraude/depot/')4

for i in file:
    prediction_vector,prediction,classes = predict_image(Path_class,str(i), model)
    if classes == 'Facture_fausse':
        #print(i)
        #name = i.split("__")[0]
        #print(i)
        name= os.path.basename(i)
        name_b =str(name).split("__")[0]
        #os.chdir('DMR_fraude/depot/')
        try :
            shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".pdf", file_destination_pixel)
            shutil.move(str(pathlib.Path('DMR_fraude/'))+"/"+name_b+".xml", file_destination_pixel)
        except:
                print("plus de factures")
    else: 
        #print(i)
        #name = i.split("__")[0]
        #print(i)
        #print(os.path.basename(i))
        #os.chdir('DMR_fraude/depot/')
        #shutil.move(name+".pdf", racine)
        #shutil.move(name+".xml", racine)
        name= os.path.basename(i)
        name_b =str(name).split("__")[0]
        #os.chdir('DMR_fraude/depot/')
        #try :
            #shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".pdf", racine)
            #shutil.move(str(pathlib.Path('DMR_fraude/depot/'))+"/"+name_b+".xml", racine)
        #except:
        print("plus de factures")

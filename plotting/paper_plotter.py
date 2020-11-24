
import os 
import sys
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path


import time

import tensorflow as tf

import math
import random
import argparse

from mycons_plot import *

import seaborn as sns



import sys 

N = 10000
N_jpg = 50000
N_hevc = 10000
M_jpeg = 21 + 1 # ( +1 ORG)
M_hevc = 27 
K = 200 

VALIDATION_DEFAULT_BYTES_SIZE = 6706179772


def construct_qp_list():
    qp_list = []
    qp_list.append(51)

    for i in range(50, 0, -2):
        qp_list.append(i)

    qp_list.append(0)
    return qp_list


def construct_qf_list():

    qp_list = []
    qp_list.append(110)

    for i in range(90, 10, -10):
        qp_list.append(i)

    qp_list.append(10)
    return qp_list

def load_np(path_to_np):
	x = np.load(path_to_np)
	return x

def verify_original_accuracy(jpeg_ranks, ranks_suffix = 'IV3'):
	
  count1 = 0
  count5 = 0
  for imgID in range(len(jpeg_ranks)):
    org_rank = int(jpeg_ranks[imgID][0])

    if org_rank <= 5:
    	count5 += 1
    	if org_rank == 1:
    		count1 += 1

  # print('DNN = %s QF = %d  Top1 accuracy = %.4f%% Top5 accuracy = %.4f%%  Compression ratio = %.4fx ' %  (ranks_suffix, 110, count1/500, count5/500, 1))


VALIDATION_DEFAULT_BYTES_SIZE = 6706179772
size_matrix    = np.load('../jpeg_sizes/imagenet_validation_sizes_110_10.npy')
def getTotalSize(jpeg_row_id, predicted_idx):
    jpeg_size = size_matrix[jpeg_row_id][predicted_idx]
    return jpeg_size


def get_QFIDX(qf):
  return int((110 - qf)/10)

def verify_qf_accuracy(jpeg_ranks, qf_index, qf, qp_list, ranks_suffix):
  
  # print(qp_list)
  # print(jpeg_ranks[0])
  # print(size_matrix[0])
  # exit(0)      
  assert(len(qp_list) == len(jpeg_ranks[0]))
  count1 = 0
  count5 = 0
  total_bytes_size = 0
  for imgID in range(len(jpeg_ranks)):
    org_rank = int(jpeg_ranks[imgID][qf_index])
    idx = get_QFIDX(qf)
    total_bytes_size += getTotalSize(imgID, idx)
    if org_rank <= 5:
      count5 += 1
      if org_rank == 1:
        count1 += 1
  			# print(org_rank)
  			# print(imgID)
  			# exit(0)

  CR  = 1.0*VALIDATION_DEFAULT_BYTES_SIZE/total_bytes_size
  print('DNN = %s QF = %d  Top1 accuracy = %.4f%% Top5 accuracy = %.4f%%  Compression ratio = %.4fx ' %  (ranks_suffix, qp_list[qf_index], count1/500, count5/500, CR))
  return qp_list[qf_index], count1/500, count5/500, CR


def ensure_model_name(x):
    try:
        x = models[x]
        return True
    except:
        return False


def generate_rank_suffixes_sep(grp_num = 0):
    lTrain = []
    lTest = []

    print(all_models)
    # exit()

    if grp_num == 0:

      lTrain.append(all_models[1])
      lTrain.append(all_models[5])
        
    
    if grp_num == 1:
      lTrain.append(all_models[4])
      lTrain.append(all_models[6])
      lTrain.append(all_models[7])


    if grp_num == 2:
      lTrain.append(all_models[3])
      lTrain.append(all_models[12])
      lTrain.append(all_models[11])
      

    if grp_num ==3 :

      lTrain.append(all_models[0])
      lTrain.append(all_models[10])
      
    
    if grp_num == 4:

      # IV3
      lTrain.append(all_models[0])
      # PNasnet
      # lTrain.append(all_models[11]


      # MobileNet V2
      lTrain.append(all_models[10])

    # Group numbers 5-10 are for generality
#     all_models             = [
#                         'IV3', 0
#                         'ResNet-V2-50', 1
#                         'Vgg16', 2
#                         'InceptionResnetV2', 3
#                         'MobileNet', 4
#                         'IV1', 5
#                         'IV4', 6
#                         'ResNet-V2-101', 7
#                         'Vgg19', 8
#                         'EfficientNet', 9
#                         'MobileNetV2', 10
#                         'Pnasnet_Large', 11
#                         'nasnet_mobile', 12
#                         'alexNet'    13

# ]
    if grp_num == 5:
      # IV4, ResNet101, MobileNet V1
      lTrain.append(all_models[6])
      lTrain.append(all_models[7])
      lTrain.append(all_models[4])


    if grp_num == 6:
      # ResNet50, IV1
      lTrain.append(all_models[1])
      lTrain.append(all_models[5])
      

      #lTrain.append(all_models[3])
      # lTrain.append(all_models[12])
      # lTrain.append(all_models[11])

    # With Pnasnet  
    if grp_num == 7:
      # PnasNet, InceptionResnetV2, nasnet_mobile
      lTrain.append(all_models[11])
      lTrain.append(all_models[3])
      lTrain.append(all_models[12])

    # Without Pnasnet
    if grp_num == 8:

        #lTrain.append(all_models[11])
        lTrain.append(all_models[3])
        lTrain.append(all_models[12])
        

    if grp_num == 9:
      # IV3, MobileNetV2
      lTrain.append(all_models[0])
      lTrain.append(all_models[10])

    # Pnasnet on its own
    if grp_num == 10:
      lTrain.append(all_models[11])
      
    

    return lTrain 



def load_ranks_for_model(ranks_suffix):
    main_path                = WORKSPACE_DIR  + '/'
    path_to_np               = os.path.join(main_path, 'ranks_' + ranks_suffix + '.npy' )
    current_ranks            = load_np(path_to_np)
    return current_ranks





def report_model_accuracies(qf_list, ranks_suffix = 'IV3'):
    jpeg_ranks = load_ranks_for_model(ranks_suffix)
    
    # print('\n\n-----> Model %s' % ranks_suffix)
    verify_original_accuracy(jpeg_ranks, ranks_suffix)


    top1_list = []
    top5_list = []
    cr_list = []
    for iqf, qf in enumerate(qf_list):
        _, top1, top5, cr = verify_qf_accuracy(jpeg_ranks, iqf, qf, qf_list, ranks_suffix)
        top1_list.append(top1)
        top5_list.append(top5)
        cr_list.append(cr)

    return top1_list, top5_list, cr_list


def report_accurcies_all_models(ranks_suffix_list, qf_list):
    top1_list = []
    top5_list = []
    cr_list   = []

    for ranks_suffix in ranks_suffix_list:
        if 'Vgg' in ranks_suffix or 'Efficient' in ranks_suffix:
            # print('Skipping these models for now %s' % ranks_suffix)
            continue
        dnn_top1, dnn_top5, dnn_cr = report_model_accuracies(qf_list, ranks_suffix)
        top1_list.append(dnn_top1)
        top5_list.append(dnn_top5)
        cr_list.append(dnn_cr)

    return top1_list, top5_list, cr_list




def quantify_gains(x_set_curve_jpeg, y_set_curve_jpeg, x_curve_selector, y_curve_selector):
    y_curve_jpeg = np.interp(x_curve_selector, x_set_curve_jpeg, y_set_curve_jpeg)
    gain         = y_curve_selector - y_curve_jpeg
    return gain, y_curve_jpeg

def load_applied_selector(selector_model, dnn, psnr, is_full):

  qfs     = QF_start[str(psnr)]
  top1_list = []
  top5_list = []
  cr_list   = []
  qfs     = QF_start[str(psnr)]
  qf_list = [10*x for x in range(int(qfs/10), 8)]

  for qf in qf_list:

    if selector_model == 'MobileNetV2':
      if is_full:
        x = np.load('../selector_vs_jpeg/apply_on_all_models/psnr' + str(psnr) +  '/MobileNetV2_Full/MobileNetV2_on_' + dnn + '_' + str(qf) + '.npy')
      else:
        x = np.load('../selector_vs_jpeg/apply_on_all_models/psnr' + str(psnr) + '/MobileNetV2_two_layer/MobileNetV2_on_' + dnn + '_' + str(qf) + '.npy')
    else:
      x = np.load('../selector_vs_jpeg/apply_on_all_models/psnr' + str(psnr) + '/IV3/IV3_on_' + dnn + '_' + str(qf) + '.npy')
    
    top1   = x[0]
    top5   = x[1]
    cr     = x[2]
    top1_list.append(top1)
    top5_list.append(top5)
    cr_list.append(cr)

  return top1_list, top5_list, cr_list




"""Makes sure the folder exists on disk.
Args:
  dir_name: Path string to the folder we want to create.
"""  
def ensure_dir_exists(dir_name):  
  if not os.path.exists(dir_name):  
    os.makedirs(dir_name)  


def plot_one_gain_array_generality(CRs, gains, marker, label, c):
    plt.plot(CRs, gains, marker, label=label, color=c, lw=2)


def plot_RA_paper_helper(top1_list, top5_list, cr_list, train_rank_suffixes, psnr, is_full, x_set_curve_jpeg, y_set_curve_jpeg,
  selector_model, rank_suffix, ax1, is_top1 = True):
  # Load the selector:
  y_set_curve_selector_top1, y_set_curve_selector_top5, x_set_curve_selector = \
  load_applied_selector(selector_model, rank_suffix, psnr, is_full)
  # load_selector(selector_model, psnr, is_full)

  # if is_full:  
  #   cur_legend =  selector_model  + 'Full @' + rank_suffix
  # else:
  #   cur_legend =  selector_model  + 'two_layers @ ' + rank_suffix

  if is_full:  
    cur_legend =  selector_model  + '_Full@' + rank_suffix
    c          = 'g'
  else:
    if selector_model == 'MobileNetV2':
      cur_legend =  selector_model  + '_2L@' + rank_suffix
      c          = 'b'
    else:
      cur_legend =  selector_model  + '_2L@' + rank_suffix
      c          = 'r'

  if is_top1:
    y = y_set_curve_selector_top1
    plot_one_gain_array_generality(x_set_curve_selector, y_set_curve_selector_top1, '-', cur_legend, c)
    if selector_model != 'MobileNetV2' or (selector_model == 'MobileNetV2' and not is_full):
      plot_one_gain_array_generality(x_set_curve_jpeg, y_set_curve_jpeg, '--'  , 'JPEG_' + selector_model, 'k')
  else:
    y = y_set_curve_selector_top5
    plot_one_gain_array_generality(x_set_curve_selector, y_set_curve_selector_top5, '-', cur_legend, c)
    if selector_model != 'MobileNetV2' or (selector_model == 'MobileNetV2' and not is_full):
      plot_one_gain_array_generality(x_set_curve_jpeg, y_set_curve_jpeg, '--', 'JPEG_' + selector_model, 'k')
  
  return x_set_curve_selector, y 


def plot_RA_curves(top1_list, top5_list, cr_list, train_rank_suffixes, psnr, grp_num = 4):

  # Top1 first
  fig = plt.figure(figsize=(15.0, 10.0)) # in inches
  ax1 = fig.add_subplot(111)
  for x_set_curve_jpeg, y_set_curve_jpeg, rank_suffix in zip(cr_list, top1_list, train_rank_suffixes):

    if(rank_suffix == 'MobileNetV2'):
      selector_model = rank_suffix

      # Plot Quantify Gains for MobileFull@MobileFull
      x_selector, y_selector = plot_RA_paper_helper(top1_list, top5_list, cr_list, 
        train_rank_suffixes, psnr, True, x_set_curve_jpeg, y_set_curve_jpeg,
        selector_model, rank_suffix, ax1, is_top1 = True)

      # Plot Quantify Gains for Mobile2@Mobile2
      x_selector, y_selector = plot_RA_paper_helper(top1_list, top5_list, cr_list, 
        train_rank_suffixes, psnr, False, x_set_curve_jpeg, y_set_curve_jpeg,
        selector_model, rank_suffix, ax1, is_top1 = True)

    elif(rank_suffix == 'IV3'):
      selector_model = rank_suffix
      
      # Plot Quantify Gains for Mobile2@Mobile2
      x_selector, y_selector = plot_RA_paper_helper(top1_list, top5_list, cr_list, 
        train_rank_suffixes, psnr, False, x_set_curve_jpeg, y_set_curve_jpeg,
        selector_model, rank_suffix, ax1, is_top1 = True)


  #plt.title('Selector Gains')
  font_size = 26
  plt.xlabel('CR', color='k', size=font_size)
  plt.ylabel('Top-1 Accuracy (%)', color='k', size=font_size)
  #plt.legend(fontsize="x-large") # using a named size
  # plt.legend(fontsize=font_size, shadow=True, loc='lower left') # using a size in points
  if psnr == 28 or psnr == 32:
    plt.legend(fontsize=font_size-2, loc='lower center', bbox_to_anchor=(0.5, 1.0),
          fancybox=True, shadow=True)
  else:
    plt.legend(fontsize=font_size, shadow=True, loc='lower left') # using a size in points

  ax1.set_xlim(min(x_selector)-0.1, max(x_selector)+0.1)
  if psnr != 26:
    qfs     = QF_start[str(psnr)]
    val = min(top1_list[1][qfs//10:-1])
    plt.gca().set_ylim(bottom=val)
  ax1.tick_params(axis='y', labelcolor='k')
  ax1.grid('on')
  plt.box('on')
  plt.xticks(size = font_size, color='k')
  plt.yticks(size = font_size, color='k')
  #fig.tight_layout()  # otherwise the right y-label is slightly clipped


  ensure_dir_exists('../final_plots/paper/' + 'Selector' + '/' + str(psnr))
  path_png = '../final_plots/paper/' + 'Selector'  + '/' + str(psnr) + '/Plot' +   '_gp' + str(grp_num) + '_Top1.pdf'
  #plt.savefig(path_png)
  #plt.show()
  plt.savefig(path_png, dpi=600, bbox_inches='tight')
  print(path_png)
    

  # Top5 second
  fig = plt.figure(figsize=(15.0, 10.0)) # in inches
  ax1 = fig.add_subplot(111)
  for x_set_curve_jpeg, y_set_curve_jpeg, rank_suffix in zip(cr_list, top5_list, train_rank_suffixes):

    if(rank_suffix == 'MobileNetV2'):
      selector_model = rank_suffix

      # Plot Quantify Gains for MobileFull@MobileFull
      x_selector, y_selector = plot_RA_paper_helper(top1_list, top5_list, cr_list, 
        train_rank_suffixes, psnr, True, x_set_curve_jpeg, y_set_curve_jpeg,
        selector_model, rank_suffix, ax1, is_top1 = False)

      # Plot Quantify Gains for Mobile2@Mobile2
      x_selector, y_selector = plot_RA_paper_helper(top1_list, top5_list, cr_list, 
        train_rank_suffixes, psnr, False, x_set_curve_jpeg, y_set_curve_jpeg,
        selector_model, rank_suffix, ax1, is_top1 = False)

    elif(rank_suffix == 'IV3'):
      selector_model = rank_suffix
      
      # Plot Quantify Gains for Mobile2@Mobile2
      x_selector, y_selector = plot_RA_paper_helper(top1_list, top5_list, cr_list, 
        train_rank_suffixes, psnr, False, x_set_curve_jpeg, y_set_curve_jpeg,
        selector_model, rank_suffix, ax1, is_top1 = False)


  #plt.title('Selector Gains')
  plt.xlabel('CR', color='k', size=font_size)
  plt.ylabel('Top-5 Accuracy (%)', color='k', size=font_size)
  #plt.legend(fontsize="x-large") # using a named size
  # plt.legend(fontsize=font_size, shadow=True, loc='lower left') # using a size in points
  if psnr == 28 or psnr == 32:
    plt.legend(fontsize=font_size-2, loc='lower center', bbox_to_anchor=(0.5, 1.0),
          fancybox=True, shadow=True)
  else:
    plt.legend(fontsize=font_size, shadow=True, loc='lower left') # using a size in points


  ax1.set_xlim(min(x_selector)-0.1, max(x_selector)+0.1)
  if psnr != 26:
    qfs     = QF_start[str(psnr)]
    val = min(top5_list[1][qfs//10:-1])
    plt.gca().set_ylim(bottom=val)
  ax1.tick_params(axis='y', labelcolor='k')
  ax1.grid('on')
  plt.box('on')
  plt.xticks(size = font_size, color='k')
  plt.yticks(size = font_size, color='k')
  #fig.tight_layout()  # otherwise the right y-label is slightly clipped


  ensure_dir_exists('../final_plots/paper/' + 'Selector' + '/' + str(psnr))
  path_png = '../final_plots/paper/' + 'Selector'  + '/' + str(psnr) + '/Plot' +   '_gp' + str(grp_num) + '_Top5.pdf'
  #plt.savefig(path_png)
  #plt.show()
  plt.savefig(path_png, dpi=600, bbox_inches='tight')
  print(path_png)
  

def plot_generality_helper(top1_list, top5_list, cr_list, train_rank_suffixes, psnr, is_full, x_set_curve_jpeg, y_set_curve_jpeg,
  selector_model, rank_suffix, ax1, is_top1 = True):
  # Load the selector:
  y_set_curve_selector_top1, y_set_curve_selector_top5, x_set_curve_selector = \
  load_applied_selector(selector_model, rank_suffix, psnr, is_full)

  # if is_full:  
  #   cur_legend =  selector_model  + 'Full @ ' + rank_suffix
  # else:
  #   cur_legend =  selector_model  + 'two_layers @ ' + rank_suffix

  # if rank_suffix == 'InceptionResnetV2':
  #   rank_suffix = 'IVRV2'

  if is_full:  
    cur_legend =  selector_model  + '@' + rank_suffix
    c          = 'g'
  else:
    if selector_model == 'MobileNetV2':
      cur_legend =  selector_model  + '-L@' + rank_suffix
      c          = 'b'
    else:
      cur_legend =  selector_model  + '-2L@' + rank_suffix
      c          = 'r'

  # Set it to None to make it more readable 
  cur_legend = None
  if is_top1:
    y = y_set_curve_selector_top1
    plot_one_gain_array_generality(x_set_curve_selector, y_set_curve_selector_top1, '-', cur_legend, c)
  else:
    y = y_set_curve_selector_top5
    plot_one_gain_array_generality(x_set_curve_selector, y_set_curve_selector_top5, '-', cur_legend, c)
   
  return x_set_curve_selector, y 



def plot_generality(top1_list, top5_list, cr_list, train_rank_suffixes, psnr, grp_num = 4):

  # Top1 first
  fig = plt.figure(figsize=(15.0, 10.0)) # in inches
  ax1 = fig.add_subplot(111)
  #for selector_model in ['IV3', 'MobileNetV2']:
  for x_set_curve_jpeg, y_set_curve_jpeg, rank_suffix in zip(cr_list, top1_list, train_rank_suffixes):
    for selector_model in ['IV3', 'MobileNetV2']:

      # Skip the cases where they are equal
      print(selector_model, rank_suffix)
      if rank_suffix == selector_model:
        continue

      if selector_model == 'MobileNetV2':
        # Plot Quantify Gains for MobileFull@MobileFull
        x_selector, y_selector = plot_generality_helper(top1_list, top5_list, cr_list, 
          train_rank_suffixes, psnr, True, x_set_curve_jpeg, y_set_curve_jpeg,
          selector_model, rank_suffix, ax1, is_top1 = True)

        # Plot Quantify Gains for Mobile2@Mobile2
        x_selector, y_selector = plot_generality_helper(top1_list, top5_list, cr_list, 
          train_rank_suffixes, psnr, False, x_set_curve_jpeg, y_set_curve_jpeg,
          selector_model, rank_suffix, ax1, is_top1 = True)


        if rank_suffix == 'InceptionResnetV2':
          t_rank_suffix = 'IVRV2'
        if rank_suffix == 'MobileNet':
          t_rank_suffix = 'MobileNetV1'
        else:
          t_rank_suffix = rank_suffix

          #rank_suffix = 'MobileNetV1'
        plot_one_gain_array_generality(x_set_curve_jpeg, y_set_curve_jpeg, '--'  ,  'Selector-JPEG@' + t_rank_suffix, 'k')

      elif selector_model == 'IV3':

         # Skip this combination
        if rank_suffix == 'MobileNetV1':
          continue
        
        # Plot Quantify Gains for IV3@Mobile2
        x_selector, y_selector = plot_generality_helper(top1_list, top5_list, cr_list, 
          train_rank_suffixes, psnr, False, x_set_curve_jpeg, y_set_curve_jpeg,
          selector_model, rank_suffix, ax1, is_top1 = True)

        if rank_suffix in 'MobileNetV2':

          if rank_suffix == 'MobileNet':
            t_rank_suffix = 'MobileNetV1'
          else:
            t_rank_suffix = rank_suffix

          if rank_suffix == 'InceptionResnetV2':
            t_rank_suffix = 'IVRV2'

          # Skip this combination
          if t_rank_suffix == 'MobileNetV1':
            continue

          # plot_one_gain_array_generality(x_set_curve_jpeg, y_set_curve_jpeg, '--'  , 'JPEG_' + rank_suffix, 'k')
          plot_one_gain_array_generality(x_set_curve_jpeg, y_set_curve_jpeg, '--'  ,  'Selector-JPEG@' + t_rank_suffix, 'k')

  #plt.title('Selector Gains')
  font_size = 26
  plt.xlabel('CR', color='k', size=font_size)
  plt.ylabel('Top-1 Accuracy (%)', color='k', size=font_size)
  #plt.legend(fontsize="x-large") # using a named size
  #if grp_num == 7:
  #plt.legend(fontsize=font_size-2, loc='lower left', ncol=3, frameon=False) # using a size in points
  # if grp_num == 9:
  #   cols = 2
  # else:
  #   cols = 3
  # plt.legend(fontsize=font_size-2, loc='lower center', bbox_to_anchor=(0.5, 1.0),
  #         ncol=cols, fancybox=True, shadow=True)
  #plt.legend(fontsize=font_size, loc='lower left', shadow=True, handletextpad=0.0, handlelength=0)

  if psnr == 28 or psnr==30 or psnr == 32:
    plt.legend(fontsize=font_size-2, loc='lower center', bbox_to_anchor=(0.5, 1.0),
          fancybox=True, shadow=True, handletextpad=0.0, handlelength=0)
  else:
    plt.legend(fontsize=font_size, shadow=True, loc='lower left', handletextpad=0.0, handlelength=0) # using a size in points

  #else:
  #plt.legend(fontsize=font_size, loc='lower left') # using a size in points

  ax1.set_xlim(min(x_selector)-0.1, max(x_selector)+0.1)
  if psnr != 26:
    qfs     = QF_start[str(psnr)]
    val = min(top1_list[-1][qfs//10:-1])
    plt.gca().set_ylim(bottom=val)
  ax1.tick_params(axis='y', labelcolor='k')
  ax1.grid('on')
  plt.box('on')
  plt.xticks(size = font_size, color = 'k')
  plt.yticks(size = font_size, color = 'k')
  #fig.tight_layout()  # otherwise the right y-label is slightly clipped


  ensure_dir_exists('../final_plots/paper/' + 'Selector' + '/' + str(psnr))
  path_png = '../final_plots/paper/' + 'Selector'  + '/' + str(psnr) + '/Plot' +   '_gp' + str(grp_num) + '_Top1.pdf'
  #plt.savefig(path_png)
  #plt.show()
  plt.savefig(path_png, dpi=600, bbox_inches='tight')
  plt.close()
  print(path_png)
    

  # Top5 second
  fig = plt.figure(figsize=(15.0, 10.0)) # in inches
  ax1 = fig.add_subplot(111)
  #for selector_model in ['IV3', 'MobileNetV2']:
  for x_set_curve_jpeg, y_set_curve_jpeg, rank_suffix in zip(cr_list, top5_list, train_rank_suffixes):
    for selector_model in ['IV3', 'MobileNetV2']:

      # Skip the cases where they are equal
      if rank_suffix == selector_model:
        continue

      if selector_model == 'MobileNetV2':
        # Plot Quantify Gains for MobileFull@MobileFull
        x_selector, y_selector = plot_generality_helper(top1_list, top5_list, cr_list, 
          train_rank_suffixes, psnr, True, x_set_curve_jpeg, y_set_curve_jpeg,
          selector_model, rank_suffix, ax1, is_top1 = False)

        # Plot Quantify Gains for Mobile2@Mobile2
        x_selector, y_selector = plot_generality_helper(top1_list, top5_list, cr_list, 
          train_rank_suffixes, psnr, False, x_set_curve_jpeg, y_set_curve_jpeg,
          selector_model, rank_suffix, ax1, is_top1 = False)

        if rank_suffix == 'InceptionResnetV2':
          t_rank_suffix = 'IVRV2'
        # plot_one_gain_array_generality(x_set_curve_jpeg, y_set_curve_jpeg, '--'  , 'JPEG_' + rank_suffix, 'k')
        if rank_suffix == 'MobileNet':
            t_rank_suffix = 'MobileNetV1'
        else:
            t_rank_suffix = rank_suffix
        plot_one_gain_array_generality(x_set_curve_jpeg, y_set_curve_jpeg, '--'  ,  'Selector-JPEG@' + t_rank_suffix, 'k')

       

      elif selector_model == 'IV3':
        
        # Plot Quantify Gains for Mobile2@Mobile2
        x_selector, y_selector = plot_generality_helper(top1_list, top5_list, cr_list, 
          train_rank_suffixes, psnr, False, x_set_curve_jpeg, y_set_curve_jpeg,
          selector_model, rank_suffix, ax1, is_top1 = False)

        if rank_suffix in 'MobileNetV2':
          if rank_suffix == 'InceptionResnetV2':
            t_rank_suffix = 'IVRV2'

          if rank_suffix == 'MobileNet':
            t_rank_suffix = 'MobileNetV1'
          else:
            t_rank_suffix = rank_suffix

          # Skip this combination
          if t_rank_suffix == 'MobileNetV1':
            continue

          # plot_one_gain_array_generality(x_set_curve_jpeg, y_set_curve_jpeg, '--'  , 'JPEG_' + rank_suffix, 'k')
          plot_one_gain_array_generality(x_set_curve_jpeg, y_set_curve_jpeg, '--'  , 'Selector-JPEG@' + t_rank_suffix, 'k')

  #plt.title('Selector Gains')
  font_size = 26
  plt.xlabel('CR', color='k', size=font_size)
  plt.ylabel('Top-5 Accuracy (%)', color='k', size=font_size)
  #plt.legend(fontsize="large") # using a named size
  #if grp_num == 7:
  #plt.legend(fontsize=font_size-2, loc='lower left', ncol=3, frameon=False) # using a size in points
  if grp_num == 9:
    cols = 2
  else:
    cols = 3
  # plt.legend(fontsize=font_size-2, loc='lower center', bbox_to_anchor=(0.5, 1.0),
  #         ncol=cols, fancybox=True, shadow=True, handletextpad=-2.0, handlelength=0)
  #plt.legend(fontsize=font_size, loc='lower left', shadow=True, handletextpad=0.0, handlelength=0)
  if psnr == 28 or psnr==30 or psnr == 32:
    plt.legend(fontsize=font_size-2, loc='lower center', bbox_to_anchor=(0.5, 1.0),
          fancybox=True, shadow=True, handletextpad=0.0, handlelength=0)
  else:
    plt.legend(fontsize=font_size, shadow=True, loc='lower left', handletextpad=0.0, handlelength=0) # using a size in points
  # ax1.annotate('Pnasnet_Large', xy=(200, 50), xycoords='axes points',
  #           size=font_size, ha='right', va='top',
  #           bbox=dict(boxstyle='round', fc='w'))
  #else:
  #plt.legend(fontsize=font_size, loc='lower left') # using a size in points
  ax1.set_xlim(min(x_selector)-0.1, max(x_selector)+0.1)
  if psnr != 26:
    qfs     = QF_start[str(psnr)]
    val = min(top5_list[-1][qfs//10:-1])
    plt.gca().set_ylim(bottom=val)
  ax1.tick_params(axis='y', labelcolor='k')
  ax1.grid('on')
  plt.box('on')
  plt.xticks(size = font_size, color = 'k')
  plt.yticks(size = font_size, color = 'k')
  #fig.tight_layout()  # otherwise the right y-label is slightly clipped


  ensure_dir_exists('../final_plots/paper/' + 'Selector' + '/' + str(psnr))
  path_png = '../final_plots/paper/' + 'Selector'  + '/' + str(psnr) + '/Plot' +   '_gp' + str(grp_num) + '_Top5.pdf'
  #plt.savefig(path_png)
  #plt.show()
  plt.savefig(path_png, dpi=600, bbox_inches='tight')
  plt.close()
  print(path_png)
  

################################
def main(_):
    # For JPEG (ORG, 100, 95, 90, ....0)
    # Anchor:
    #path_to_np =  '/home/h2amer/work/workspace/inception_selector_training/train/jpeg_ranks/ranks_IV3.npy'
    #ranks     = load_np(path_to_np)

    
    # Constrcut the lsit
    qf_list = construct_qf_list()

    
    is_RA_curves = False
    # Generate the rank suffixes

    # grp_num = 0
    #for psnr in [26, 28 , 30 , 32]:
    for psnr in [26]:
    # for psnr in [28]:
      # for grp_num in [0,1,2,3]:
      if is_RA_curves:
        for grp_num in [4]:
          train_rank_suffixes                     = generate_rank_suffixes_sep(grp_num=grp_num)
          top1_list, top5_list, cr_list           = report_accurcies_all_models(train_rank_suffixes, qf_list)
          # Plot Gains for Paper (Mobile@Mobile) and IV3@IV3
          plot_RA_curves(top1_list, top5_list, cr_list, train_rank_suffixes, psnr = psnr, grp_num = grp_num)
      else:
        # for grp_num in [5, 7]:
        for grp_num in range(5, 10 + 1):
          train_rank_suffixes                     = generate_rank_suffixes_sep(grp_num=grp_num)
          top1_list, top5_list, cr_list           = report_accurcies_all_models(train_rank_suffixes, qf_list)

          # Plot Gains For Paper Generality 
          plot_generality(top1_list, top5_list, cr_list, train_rank_suffixes, psnr = psnr, grp_num = grp_num)

    

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  

    parser.add_argument(  
      '--select',  
      type=str,  
      default='jpeg',  
      help='select to Save which '  
  )

    parser.add_argument(  
      '--START',  
      type=int,  
      default='1',  
      help='start of the sequence  '  
  )

    parser.add_argument(  
      '--END',  
      type=int,  
      default='50000',  
      help='end of the sequence '  
  )

    parser.add_argument(  
      '--model_name',  
      type=str,  
      default='IV1',  
      help='model name'  
  )

    FLAGS, unparsed = parser.parse_known_args()  


tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


from saliency_toolbox import calculate_measures
# sm_dir = '/hy-tmp/MLMSNet/test_results/ECSSD/mask/'
# gt_dir = '/hy-tmp/MLMSNet/sal_datasets/ECSSD/gt/'
# sm_dir = '/hy-tmp/MLMSNet/test_results/THUR/mask/'
# gt_dir = '/hy-tmp/MLMSNet/sal_datasets/THUR/gt/'
#dataset_names = ['PASCAL-S','ECSSD']
dataset_names = ['HKU-IS','OMRON','SOD','DUT-test','PASCAL-S','ECSSD']
#for dataset in dataset_names:
   # sm_dir = '/hy-tmp/MLMSNet/test_results/'+dataset+'/mask/'
for dataset_name in dataset_names:
  sm_base = '/hy-tmp/MLMSNet/test_results_b3_base_ed1_mlm1_v2_v0407/'+dataset_name+'/mask/'
  gt_dir = '/hy-tmp/MLMSNet/sal_datasets/'+dataset_name+'/gt/'
  # sm_dir = '/hy-tmp/MLMSNet/test_results/omron/mask/'
  # gt_dir = '/hy-tmp/MLMSNet/sal_datasets/OMRON/gt/'
  # sm_dir = '/hy-tmp/MLMSNet/test_results/sod/mask/'
  # gt_dir = '/hy-tmp/MLMSNet/sal_datasets/SOD/gt'
  # sm_dir = 'SM/'
  # gt_dir = 'GT/'
  sup_res   = calculate_measures(gt_dir, sm_base, ['MAE', 'S-measure','Adp-F'], save=dataset_name+'_v0405')
  fd = open("new_3b_base_1ed_1mlm_v2___9.txt",'a+')

  fd.write(dataset_name +' mae : '+ str(sup_res['MAE'])+' S-measure : '+str(sup_res['S-measure'])+' Adp-F : '+str(sup_res['Adp-F'])+'\n')
  fd.close()
  print(dataset_name, ' mae : ', sup_res['MAE'], 'S-measure : ',sup_res['S-measure'],'Adp-F : ',sup_res['Adp-F'])
  
    #  base_res = calculate_measures(gt_dir, sm_base, ['MAE','E-measure', 'S-measure','Wgt-F','Max-F','Adp-F'], save='sod')
  #   fe = open("eval_base.txt",'a+')

    #  fe.write(dataset +' mae : '+ str(base_res['MAE'])+ ' E-measure : ' +str(base_res['E-measure'])+' S-measure : '+str(base_res['S-measure'])+' Wgt-F : '+str(base_res['Wgt-F'])+' Adp-F : '+str(base_res['Adp-F'])+'\n')
    #  fe.close()

  #duts   mae :  0.04568197 E-measure :  0.899033262928064  S-measure :  0.8489085816334003 Wgt-F :  0.7943019322290533 Adp-F :  0.8145171523971444
  #ecssd  mae :  0.03629435 E-measure :  0.9472822180484138 S-measure :  0.9059146350208593 Wgt-F :  0.9028640202251393 Adp-F :  0.9166406972249822
  #omron  mae :  0.05922905 E-measure :  0.8530007209406391 S-measure :  0.8102967024018489 Wgt-F :  0.7260727318185218 Adp-F :  0.7485040667500404
  #sod mae :  0.10423665 E-measure :  0.8139244693841753 S-measure :  0.7727241237011658 Wgt-F :  0.7471805063754822 Adp-F :  0.8029934943124942
  #hkuis mae :  0.031145234 E-measure :  0.9503135651460997 S-measure :  0.8995645735612825 Wgt-F :  0.8915280447698808 Adp-F :  0.9041993770739867
  
  ###v321
  #PASCAL-S  mae :  0.11748628 E-measure :  0.8328014962974676 S-measure :  0.7813361932279799 Wgt-F :  0.7813294862262729 Adp-F :  0.8265609776688044
  #DUT-test  mae :  0.04404416 E-measure :  0.9050440272765882 S-measure :  0.8525624353569886 Wgt-F :  0.8035924893910211 Adp-F :  0.8232688350197555
  #ECSSD     mae :  0.037010927 E-measure :  0.9444958636613956 S-measure :  0.903340654456433 Wgt-F :  0.9007830159342797 Adp-F :  0.9161483203562129
  #THUR      mae :  0.07384023 E-measure :  0.8436017226575564 S-measure :  0.8115318300859691 Wgt-F :  0.720132282034741 Adp-F :  0.7387307016260297
  #SOD       mae :  0.11024639 E-measure :  0.8060584683485804 S-measure :  0.7667445151671889 Wgt-F :  0.7394113352512389 Adp-F :  0.7942230068140005
  #OMRON     mae :  0.060021527 E-measure :  0.8498925860708846 S-measure :  0.8070597859045221 Wgt-F :  0.7218584585459576 Adp-F :  0.7446085584291255
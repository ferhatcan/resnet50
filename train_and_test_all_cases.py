from Dataset import Dataset
import cifar_DataCreate as cifar
import transfer_learning as tl

"""
# tl.resnet50_test('whole')
print('*'*25)
print('Starts Trainings\n')

savepath_whole_ccl, savename_whole_ccl          = tl.resnet50_trainval('whole', 'cross_entropy_loss')
savepath_only_fc_ccl, savename_only_fc_ccl      = tl.resnet50_trainval('only_fc', 'cross_entropy_loss')
savepath_fc_layer4_ccl, savename_fc_layer4_ccl  = tl.resnet50_trainval('fc+layer4', 'cross_entropy_loss')
savepath_whole_hl, savename_whole_hl            = tl.resnet50_trainval('whole', 'hinge_loss')
savepath_only_fc_hl, savename_only_fc_hl        = tl.resnet50_trainval('only_fc', 'hinge_loss')
savepath_fc_layer4_hl, savename_fc_layer4_hl    = tl.resnet50_trainval('fc+layer4', 'hinge_loss')

print('Starts Testing\n')

test_acc_whole_ccl, test_acc_top2_whole_ccl         = tl.resnet50_test(savepath_whole_ccl, savename_whole_ccl)
test_acc_only_fc_ccl, test_acc_top2_only_fc_ccl     = tl.resnet50_test(savepath_only_fc_ccl, savename_only_fc_ccl)
test_acc_fc_layer4_ccl, test_acc_top2_fc_layer4_ccl = tl.resnet50_test(savepath_fc_layer4_ccl, savename_fc_layer4_ccl)
test_acc_whole_hl, test_acc_top2_whole_hl           = tl.resnet50_test(savepath_whole_hl, savename_whole_hl)
test_acc_only_fc_hl, test_acc_top2_only_fc_hl       = tl.resnet50_test(savepath_only_fc_hl, savename_only_fc_hl)
test_acc_fc_layer4_hl, test_acc_top2_fc_layer4_hl   = tl.resnet50_test(savepath_fc_layer4_hl, savename_fc_layer4_hl)


print('*'*25)
print('End')
"""

"""
print('*'*25)
print('Starts Trainings\n')

savepath_whole_ccl, savename_whole_ccl          = tl.resnet50_trainval('whole', 'cross_entropy_loss', size=0.1)
savepath_only_fc_ccl, savename_only_fc_ccl      = tl.resnet50_trainval('only_fc', 'cross_entropy_loss', size=0.1)
savepath_fc_layer4_ccl, savename_fc_layer4_ccl  = tl.resnet50_trainval('fc+layer4', 'cross_entropy_loss', size=0.1)
savepath_whole_hl, savename_whole_hl            = tl.resnet50_trainval('whole', 'hinge_loss', size=0.1)
savepath_only_fc_hl, savename_only_fc_hl        = tl.resnet50_trainval('only_fc', 'hinge_loss', size=0.1)
savepath_fc_layer4_hl, savename_fc_layer4_hl    = tl.resnet50_trainval('fc+layer4', 'hinge_loss', size=0.1)

print('Starts Testing\n')

test_acc_whole_ccl, test_acc_top2_whole_ccl         = tl.resnet50_test(savepath_whole_ccl, savename_whole_ccl, size=1)
test_acc_only_fc_ccl, test_acc_top2_only_fc_ccl     = tl.resnet50_test(savepath_only_fc_ccl, savename_only_fc_ccl, size=0.1)
test_acc_fc_layer4_ccl, test_acc_top2_fc_layer4_ccl = tl.resnet50_test(savepath_fc_layer4_ccl, savename_fc_layer4_ccl, size=0.1)
test_acc_whole_hl, test_acc_top2_whole_hl           = tl.resnet50_test(savepath_whole_hl, savename_whole_hl, size=0.1)
test_acc_only_fc_hl, test_acc_top2_only_fc_hl       = tl.resnet50_test(savepath_only_fc_hl, savename_only_fc_hl, size=0.1)
test_acc_fc_layer4_hl, test_acc_top2_fc_layer4_hl   = tl.resnet50_test(savepath_fc_layer4_hl, savename_fc_layer4_hl, size=0.1)


print('*'*25)
print('End')
"""

print('*'*25)
print('Starts Trainings\n')

savepath_whole_ccl, savename_whole_ccl          = tl.resnet50_trainval('whole', 'cross_entropy_loss', size=0.01)
savepath_only_fc_ccl, savename_only_fc_ccl      = tl.resnet50_trainval('only_fc', 'cross_entropy_loss', size=0.01)
savepath_fc_layer4_ccl, savename_fc_layer4_ccl  = tl.resnet50_trainval('fc+layer4', 'cross_entropy_loss', size=0.01)
savepath_whole_hl, savename_whole_hl            = tl.resnet50_trainval('whole', 'hinge_loss', size=0.01)
savepath_only_fc_hl, savename_only_fc_hl        = tl.resnet50_trainval('only_fc', 'hinge_loss', size=0.01)
savepath_fc_layer4_hl, savename_fc_layer4_hl    = tl.resnet50_trainval('fc+layer4', 'hinge_loss', size=0.01)

print('Starts Testing\n')

test_acc_whole_ccl, test_acc_top2_whole_ccl         = tl.resnet50_test(savepath_whole_ccl, savename_whole_ccl, size=1)
test_acc_only_fc_ccl, test_acc_top2_only_fc_ccl     = tl.resnet50_test(savepath_only_fc_ccl, savename_only_fc_ccl, size=1)
test_acc_fc_layer4_ccl, test_acc_top2_fc_layer4_ccl = tl.resnet50_test(savepath_fc_layer4_ccl, savename_fc_layer4_ccl, size=1)
test_acc_whole_hl, test_acc_top2_whole_hl           = tl.resnet50_test(savepath_whole_hl, savename_whole_hl, size=1)
test_acc_only_fc_hl, test_acc_top2_only_fc_hl       = tl.resnet50_test(savepath_only_fc_hl, savename_only_fc_hl, size=1)
test_acc_fc_layer4_hl, test_acc_top2_fc_layer4_hl   = tl.resnet50_test(savepath_fc_layer4_hl, savename_fc_layer4_hl, size=1)


print('*'*25)
print('End')
import matplotlib.pyplot as plt
import numpy as np

adabf_url = {'one-shot': [[202,231,199,219,228], 
                        [128,181,114,112,129], 
                        [81,67,75,63,85], 
                        [46,43,33,38,41], 
                        [20,44,30,29,37]], 
            'unif': [[802430,839907,845930,817712,820459], 
                     [335544,333474,338278,339517,337305], 
                     [204160,206922,203622,203678,207596], 
                     [153486,148655,145415,144221,146676], 
                     [110771,108895,105701,110929,109307]], 
            'zipf': [[219114,147157,403366,200582,358291], 
                     [68781,67148,125640,58392,71245], 
                     [77730,75837,47912,52137,56946], 
                     [55699,54746,50508,48161,50648], 
                     [16272,18526,46522,40050,68436]]}
adabf_news = {'one-shot': [[10,8,12,7,9], 
                        [4,11,13,11,7], 
                        [6,4,8,10,3], 
                        [3,3,4,8,4], 
                        [3,1,4,1,7]], 
            'unif': [[7031,1656,2285,2228,5392], 
                     [1726,1452,1443,2283,1428], 
                     [864,2007,555,1471,877], 
                     [1452,288,288,594,288], 
                     [548,594,2301,1172,2009]], 
            'zipf': [[77,73,73,78,167], 
                     [73,1,77,139,6], 
                     [1,73,1,73,77], 
                     [1,78,73,73,73], 
                     [77,1,77,73,1]]}
adabf_ember = {'one-shot': [[4668,4553,4627,4637,4720],
                            [3673,3763,3651,3654,3690],
                            [2872,2856,2852,2980,2906],
                            [2370,2363,2430,2368,2401],
                            [2048,1966,2018,1968,1947]],
               'unif': [[41613,39954,41141,41245,41368],
                        [33500,33085,33464,32986,32984],
                        [27412,26558,26635,26639,26597],
                        [21763,23518,21666,21973,21268],
                        [18283,17937,17877,17582,17291]], 
               'zipf': [[20954,21520,24555,17642,17465],
                        [23880,17079,21111,17165,19631],
                        [19863,14537,14891,31437,35154],
                        [11311,943,589,1314,26315],
                        [40165,11672,4392,4737,1115,10916]]}

plbf_url = {'one-shot': [[176], 
                        [105], 
                        [73], 
                        [30], 
                        [20]], 
            'unif': [[10970], 
                     [6563], 
                     [4456], 
                     [1821], 
                     [1257]], 
            'zipf': [[126], 
                     [6630], 
                     [45], 
                     [25837], 
                     [56]]}
plbf_news = {'one-shot': [[7], 
                        [5], 
                        [4], 
                        [3], 
                        [3]], 
            'unif': [[1931], 
                     [1342], 
                     [1124], 
                     [821], 
                     [821]], 
            'zipf': [[72], 
                     [72], 
                     [72], 
                     [72], 
                     [72]]}
plbf_ember = {'one-shot': [[4754], 
                        [3850], 
                        [3019], 
                        [2545], 
                        [1945]], 
            'unif': [[59654], 
                     [48392], 
                     [37626], 
                     [31965], 
                     [24575]], 
            'zipf': [[32523], 
                     [39466], 
                     [15066], 
                     [26068], 
                     [4364]]}

aqf_url = {'one-shot': [[251,257,274,273,285], 
                        [130,133,137,143,154], 
                        [75,79,74,72,79], 
                        [38,39,41,41,40], 
                        [15,16,15,16,15]], 
            'unif': [[10745,10824,10771,10749,10712,], 
                     [5670,5723,5659,5746,5702], 
                     [2959,2887,2886,2878,2854], 
                     [1441,1419,1458,1447,1407], 
                     [706,704,695,746,711]], 
            'zipf': [[10820,10729,10651,10773,10693], 
                     [5687,5747,5712,5680,5672], 
                     [2937,2873,2961,2946,2924], 
                     [1468,1397,1490,1441,1435], 
                     [726,714,714,719,714]]}
aqf_news = {'one-shot': [[55,72,45,49,82], 
                        [22,36,39,37,34], 
                        [16,13,13,12,12], 
                        [10,10,8,9,3], 
                        [3,11,11,4,1]], 
            'unif': [[8528,8390,8530,8508,8476], 
                     [5056,5033,4946,5051,5050], 
                     [2669,2801,2710,2692,2718], 
                     [1402,1401,1380,1370,1415], 
                     [687,714,695,716,717]], 
            'zipf': [[8535,8423,8457,8503,8396], 
                     [4997,4980,4972,5052,5026], 
                     [2728,2745,2740,2672,2716], 
                     [1382,1348,1401,1391,1388], 
                     [730,720,675,700,717]]}
aqf_ember = {'one-shot': [[2328,2288,2279,2300,2285], 
                        [1180,1175,1119,1140,1172], 
                        [571,613,571,535,576], 
                        [288,308,267,304,267], 
                        [151,138,160,133,150]], 
            'unif': [[21706,21922,21892,21690,21852], 
                     [11176,11135,11005,11129,11103], 
                     [5597,5531,5588,5608,5619], 
                     [2815,2816,2855,2808,2815], 
                     [1391,1378,1369,1384,1409]], 
            'zipf': [[21850,21882,21780,21820,21901], 
                     [11084,11051,11081,11155,11048], 
                     [5540,5577,5532,5596,5590], 
                     [2829,2834,2802,2833,2785], 
                     [1300,1350,1315,1418,1403]]}

num_rows = {'url': 162798, 'news': 35919, 'ember': 800000}
num_true_negative_url = {'one-shot': 107117, 'unif': 6580291, 'zipf': 2292307}
num_true_negative_news = {'one-shot': 17122, 'unif': 4767929, 'zipf': 7707754}
num_true_negative_ember = {'one-shot': 400000, 'unif': 5001438, 'zipf': 2518122}

url_filter_sizes = [338400, 371808, 405216, 438624, 472032]
news_filter_sizes = [86328, 94840, 103352, 111864, 120376]
ember_filter_sizes = [1340208, 1472560, 1604912, 1737264, 1869616]

# plot url data ----------------------------------------------------------------------------------------------------------


plt.plot(url_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_url['one-shot']) for i in plbf_url['one-shot']],  label='PLBF++', color="blue")
plt.plot(url_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_url['one-shot']) for i in aqf_url['one-shot']], label='AQF', color="orange")
aqf_url_fprs = [[i / (float)(i + num_true_negative_url['one-shot'])for i in array] for array in aqf_url['one-shot']]
aqf_url_25 = [np.percentile(i, 25) for i in aqf_url_fprs]
aqf_url_75 = [np.percentile(i, 75) for i in aqf_url_fprs]
plt.fill_between(url_filter_sizes, aqf_url_25, aqf_url_75, alpha=0.2, color="orange")
plt.plot(url_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_url['one-shot']) for i in adabf_url['one-shot']], label='ADA-BF', color="green")
adabf_url_fprs = [[i / (float)(i + num_true_negative_url['one-shot'])for i in array] for array in adabf_url['one-shot']]
adabf_url_25 = [np.percentile(i, 25) for i in adabf_url_fprs]
adabf_url_75 = [np.percentile(i, 75) for i in adabf_url_fprs]
plt.fill_between(url_filter_sizes, adabf_url_25, adabf_url_75, alpha=0.2, color="green")

plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('Median FPR-Space Tradeoff on URLs (One-pass)')
plt.legend()
plt.savefig('URL_one_shot.pdf', bbox_inches='tight')
plt.clf()

plt.plot(url_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_url['unif']) for i in plbf_url['unif']], label='PLBF++', color="blue")
plt.plot(url_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_url['unif']) for i in aqf_url['unif']], label='AQF', color="orange")
aqf_url_fprs = [[i / (float)(i + num_true_negative_url['unif'])for i in array] for array in aqf_url['unif']]
aqf_url_25 = [np.percentile(i, 25) for i in aqf_url_fprs]
aqf_url_75 = [np.percentile(i, 75) for i in aqf_url_fprs]
plt.fill_between(url_filter_sizes, aqf_url_25, aqf_url_75, alpha=0.2, color="orange")
plt.plot(url_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_url['unif']) for i in adabf_url['unif']], label='ADA-BF', color="green")
adabf_url_fprs = [[i / (float)(i + num_true_negative_url['unif'])for i in array] for array in adabf_url['unif']]
adabf_url_25 = [np.percentile(i, 25) for i in adabf_url_fprs]
adabf_url_75 = [np.percentile(i, 75) for i in adabf_url_fprs]
plt.fill_between(url_filter_sizes, adabf_url_25, adabf_url_75, alpha=0.2, color="green")
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('Median FPR-Space Tradeoff on URLs (10M Uniform)')
plt.legend()
plt.savefig('URL_10M_unif.pdf', bbox_inches='tight')
plt.clf()

plt.plot(url_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_url['zipf']) for i in plbf_url['zipf']], label='PLBF++', color="blue")
plt.plot(url_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_url['zipf']) for i in aqf_url['zipf']], label='AQF', color="orange")
aqf_url_fprs = [[i / (float)(i + num_true_negative_url['zipf'])for i in array] for array in aqf_url['zipf']]
aqf_url_25 = [np.percentile(i, 25) for i in aqf_url_fprs]
aqf_url_75 = [np.percentile(i, 75) for i in aqf_url_fprs]
plt.fill_between(url_filter_sizes, aqf_url_25, aqf_url_75, alpha=0.2, color="orange")
plt.plot(url_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_url['zipf']) for i in adabf_url['zipf']], label='ADA-BF', color="green")
adabf_url_fprs = [[i / (float)(i + num_true_negative_url['zipf'])for i in array] for array in adabf_url['zipf']]
adabf_url_25 = [np.percentile(i, 25) for i in adabf_url_fprs]
adabf_url_75 = [np.percentile(i, 75) for i in adabf_url_fprs]
plt.fill_between(url_filter_sizes, adabf_url_25, adabf_url_75, alpha=0.2, color="green")
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('Median FPR-Space Tradeoff on URLs (10M Zipfian)')
plt.legend()
plt.savefig('URL_10M_zipf.pdf', bbox_inches='tight')
plt.clf()

# plot news data ----------------------------------------------------------------------------------------------------------

plt.plot(news_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_news['one-shot']) for i in plbf_news['one-shot']],  label='PLBF++', color="blue")
plt.plot(news_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_news['one-shot']) for i in aqf_news['one-shot']], label='AQF', color="orange")
aqf_news_fprs = [[i / (float)(i + num_true_negative_news['one-shot'])for i in array] for array in aqf_news['one-shot']]
aqf_news_25 = [np.percentile(i, 25) for i in aqf_news_fprs]
aqf_news_75 = [np.percentile(i, 75) for i in aqf_news_fprs]
plt.fill_between(news_filter_sizes, aqf_news_25, aqf_news_75, alpha=0.2, color="orange")
plt.plot(news_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_news['one-shot']) for i in adabf_news['one-shot']], label='ADA-BF', color="green")
adabf_news_fprs = [[i / (float)(i + num_true_negative_news['one-shot'])for i in array] for array in adabf_news['one-shot']]
adabf_news_25 = [np.percentile(i, 25) for i in adabf_news_fprs]
adabf_news_75 = [np.percentile(i, 75) for i in adabf_news_fprs]
plt.fill_between(news_filter_sizes, adabf_news_25, adabf_news_75, alpha=0.2, color="green")
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('Median FPR-Space Tradeoff on News (One-pass)')
plt.legend()
plt.savefig('News_one_shot.pdf', bbox_inches='tight')
plt.clf()

plt.plot(news_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_news['unif']) for i in plbf_news['unif']], label='PLBF++', color="blue")
plt.plot(news_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_news['unif']) for i in aqf_news['unif']], label='AQF', color="orange")
aqf_news_fprs = [[i / (float)(i + num_true_negative_news['unif'])for i in array] for array in aqf_news['unif']]
aqf_news_25 = [np.percentile(i, 25) for i in aqf_news_fprs]
aqf_news_75 = [np.percentile(i, 75) for i in aqf_news_fprs]
plt.fill_between(news_filter_sizes, aqf_news_25, aqf_news_75, alpha=0.2, color="orange")
plt.plot(news_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_news['unif']) for i in adabf_news['unif']], label='ADA-BF', color="green")
adabf_news_fprs = [[i / (float)(i + num_true_negative_news['unif'])for i in array] for array in adabf_news['unif']]
adabf_news_25 = [np.percentile(i, 25) for i in adabf_news_fprs]
adabf_news_75 = [np.percentile(i, 75) for i in adabf_news_fprs]
plt.fill_between(news_filter_sizes, adabf_news_25, adabf_news_75, alpha=0.2, color="green")
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('Median FPR-Space Tradeoff on News (10M Uniform)')
plt.legend()
plt.savefig('News_10M_unif.pdf', bbox_inches='tight')
plt.clf()

plt.plot(news_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_news['zipf']) for i in plbf_news['zipf']], label='PLBF++', color="blue")
plt.plot(news_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_news['zipf']) for i in aqf_news['zipf']], label='AQF', color="orange")
aqf_news_fprs = [[i / (float)(i + num_true_negative_news['zipf'])for i in array] for array in aqf_news['zipf']]
aqf_news_25 = [np.percentile(i, 25) for i in aqf_news_fprs]
aqf_news_75 = [np.percentile(i, 75) for i in aqf_news_fprs]
plt.fill_between(news_filter_sizes, aqf_news_25, aqf_news_75, alpha=0.2, color="orange")
plt.plot(news_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_news['zipf']) for i in adabf_news['zipf']], label='ADA-BF', color="green")
adabf_news_fprs = [[i / (float)(i + num_true_negative_news['zipf'])for i in array] for array in adabf_news['zipf']]
adabf_news_25 = [np.percentile(i, 25) for i in adabf_news_fprs]
adabf_news_75 = [np.percentile(i, 75) for i in adabf_news_fprs]
plt.fill_between(news_filter_sizes, adabf_news_25, adabf_news_75, alpha=0.2, color="green")
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('Median FPR-Space Tradeoff on News (10M Zipfian)')
plt.legend()
plt.savefig('News_10M_zipf.pdf', bbox_inches='tight')
plt.clf()

# plot ember data ----------------------------------------------------------------------------------------------------------

plt.plot(ember_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_ember['one-shot']) for i in plbf_ember['one-shot']],  label='PLBF++', color="blue")
plt.plot(ember_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_ember['one-shot']) for i in aqf_ember['one-shot']], label='AQF', color="orange")
aqf_ember_fprs = [[i / (float)(i + num_true_negative_ember['one-shot'])for i in array] for array in aqf_ember['one-shot']]
aqf_ember_25 = [np.percentile(i, 25) for i in aqf_ember_fprs]
aqf_ember_75 = [np.percentile(i, 75) for i in aqf_ember_fprs]
plt.fill_between(ember_filter_sizes, aqf_ember_25, aqf_ember_75, alpha=0.2, color="orange")
plt.plot(ember_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_ember['one-shot']) for i in adabf_ember['one-shot']], label='ADA-BF', color="green")
adabf_ember_fprs = [[i / (float)(i + num_true_negative_ember['one-shot'])for i in array] for array in adabf_ember['one-shot']]
adabf_ember_25 = [np.percentile(i, 25) for i in adabf_ember_fprs]
adabf_ember_75 = [np.percentile(i, 75) for i in adabf_ember_fprs]
plt.fill_between(ember_filter_sizes, adabf_ember_25, adabf_ember_75, alpha=0.2, color="green")
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('Median FPR-Space Tradeoff on Ember (One-pass)')
plt.legend()
plt.savefig('ember_one_shot.pdf', bbox_inches='tight')
plt.clf()

plt.plot(ember_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_ember['unif']) for i in plbf_ember['unif']], label='PLBF++', color="blue")
plt.plot(ember_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_ember['unif']) for i in aqf_ember['unif']], label='AQF', color="orange")
aqf_ember_fprs = [[i / (float)(i + num_true_negative_ember['unif'])for i in array] for array in aqf_ember['unif']]
aqf_ember_25 = [np.percentile(i, 25) for i in aqf_ember_fprs]
aqf_ember_75 = [np.percentile(i, 75) for i in aqf_ember_fprs]
plt.fill_between(ember_filter_sizes, aqf_ember_25, aqf_ember_75, alpha=0.2, color="orange")
plt.plot(ember_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_ember['unif']) for i in adabf_ember['unif']], label='ADA-BF', color="green")
adabf_ember_fprs = [[i / (float)(i + num_true_negative_ember['unif'])for i in array] for array in adabf_ember['unif']]
adabf_ember_25 = [np.percentile(i, 25) for i in adabf_ember_fprs]
adabf_ember_75 = [np.percentile(i, 75) for i in adabf_ember_fprs]
plt.fill_between(ember_filter_sizes, adabf_ember_25, adabf_ember_75, alpha=0.2, color="green")
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('Median FPR-Space Tradeoff on Ember (10M Uniform)')
plt.legend()
plt.savefig('ember_10M_unif.pdf', bbox_inches='tight')
plt.clf()

plt.plot(ember_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_ember['zipf']) for i in plbf_ember['zipf']], label='PLBF++', color="blue")
plt.plot(ember_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_ember['zipf']) for i in aqf_ember['zipf']], label='AQF', color="orange")
aqf_ember_fprs = [[i / (float)(i + num_true_negative_ember['zipf'])for i in array] for array in aqf_ember['zipf']]
aqf_ember_25 = [np.percentile(i, 25) for i in aqf_ember_fprs]
aqf_ember_75 = [np.percentile(i, 75) for i in aqf_ember_fprs]
plt.fill_between(ember_filter_sizes, aqf_ember_25, aqf_ember_75, alpha=0.2, color="orange")
plt.plot(ember_filter_sizes, [np.median(i) / (float)(np.median(i) + num_true_negative_ember['zipf']) for i in adabf_ember['zipf']], label='ADA-BF', color="green")
adabf_ember_fprs = [[i / (float)(i + num_true_negative_ember['zipf'])for i in array] for array in adabf_ember['zipf']]
adabf_ember_25 = [np.percentile(i, 25) for i in adabf_ember_fprs]
adabf_ember_75 = [np.percentile(i, 75) for i in adabf_ember_fprs]
plt.fill_between(ember_filter_sizes, adabf_ember_25, adabf_ember_75, alpha=0.2, color="green")
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('Median FPR-Space Tradeoff on Ember (10M Zipfian)')
plt.legend()
plt.savefig('ember_10M_zipf.pdf', bbox_inches='tight')
plt.clf()
import ch5_1 as prepIris
import ch5_2 as breastCancer
import ch5_3 as prepMnist
import ch5_4 as cifar10
import ch5_7 as augmentCifar
import concat_images as concatinateAugmented

# labels = 0
# features = 1
# data = prepIris.importData()
# data = prepIris.randomize(data[labels], data[features])
# prepIris.save(data[labels], data[features])
# # second feature has some outliers, but since the the features 
# # all have similar scales we'll use the features as they are
# prepIris.plot(data[features])

# breastCancer.run()
# prepMnist.run()
# cifar10.run()
# augmentCifar.main()
concatinateAugmented.main()
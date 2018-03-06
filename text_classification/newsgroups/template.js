/**
 * In this section, you will modify a standard bag-of-words model to try and achieve 
 * better performance. The current implementation uses a single-layer neural network 
 * which is equivalent to multiclass logistic regression.
 * 
 * You should modify it by adding additional hidden layers, trying different 
 * regularization methods (dropout, weight decay, etc.). Alternatively, you can load 
 * GloVe word embeddings with the GloveEmbedding helper class. You should include the results - both positive and negative - 
 * for all the models you tried.
 * 
 * If you prefer, you can also complete this assignment offline with your standard 
 * IDE by modifying cloning the below repository and modifying template.js.
 *
 *   > https://github.com/mintingle/rnn-6s198/tree/master/text_classification/newsgroups 
 * 
 */
buildModel = (dataset) => {
    // misc. hyperparmeters
    // TODO: add your parameters (optional) for dropout, weight decay, etc. here
    const batch_size = 32;

    // construct the computation graph
    const g = new deeplearn.Graph();
    const math = deeplearn.ENV.math;

    // create placeholders for the inputs
    const inputTensor = g.placeholder('input', [dataset.input_dims]);
    const labelTensor = g.placeholder('label', [be.label_dims]);

    // add a single feedforward layer
    // TODO: try multiple layers, or other modifications
    const outputTensor = g.layers.dense('dense_layer', inputTensor, dataset.output_dims, undefined, true);

    // loss functions
    const costTensor = g.softmaxCrossEntropyCost(outputTensor, labelTensor);
    const accTensor = g.argmaxEquals(outputTensor, labelTensor)

    // set up a new session
    const session = new deeplearn.Session(g, math);
    const optimizer = new deeplearn.SGDOptimizer(0.01); // Replace this with AdamOptimizer and bad things (TM) start to happen...

    // every epoch runs train and test and prints the loss/acc to console
    const run_epoch = (epoch) => {
        math.scope(() => {
            testAcc = () => {
                const accs = [];
                const losses = [];
                for (let i = 0; i < dataset.nb_test; i++) {
                    let res = session.evalAll([accTensor, costTensor], [
                        {tensor: inputTensor, data: dataset.testX},
                        {tensor: labelTensor, data: dataset.testY}
                    ]);
                    accs.push(res[0].get())
                    losses.push(res[1].get())
                }
                return JSON.stringify({
                    "acc": average(accs),
                    "loss": average(losses)
                })
            }
            
            trainLoss = () => {
                let cost = []
                for (let batch = 0; batch < dataset.nb_train / batch_size; batch++)
                    cost.push(session.train(costTensor, [
                        {tensor: inputTensor, data: dataset.trainX},
                        {tensor: labelTensor, data: dataset.trainY}
                    ], batch_size, optimizer, deeplearn.CostReduction.MEAN).get());
                return average(cost)
            }

            console.log("Epoch " + epoch + ", Train Loss: " + trainLoss() + ", Test Loss: " + testAcc())
        })
    }

    // each epoch takes ~10 seconds
    for (let i = 0; i < 1; i++)
        run_epoch(i)
}

// download the dataset, the helper classes BagOfWordsEmbedding and BinaryEncoder can
// be found at: https://courses.csail.mit.edu/6.s198/spring-2018/rnn-assignment/text_classification/newsgroups/helpers.js
$.get("data/newsgroups.json", (dataset) => {
    // keep track of the number of test/train samples
    dataset.nb_test = dataset.testY.length
    dataset.nb_train = dataset.trainY.length

    // Use a bag of words embedding for the text
    bowe = new BagOfWordsEmbedding()
    bowe.fit(dataset.trainX)
    dataset.input_dims = bowe.word_dims
    dataset.trainX = bowe.transform(dataset.trainX)
    dataset.testX = bowe.transform(dataset.testX)

    // Use a one-hot binary encoding for the prediction target
    be = new BinaryEncoder()
    be.fit(dataset.trainY)
    dataset.output_dims = be.label_dims
    dataset.trainY = be.transform(dataset.trainY)
    dataset.testY = be.transform(dataset.testY)

    // Use deeplearnjs's data providers
    const trainProvider = new deeplearn.InCPUMemoryShuffledInputProviderBuilder([dataset.trainX, dataset.trainY])
    dataset.trainX = trainProvider.getInputProvider(0)
    dataset.trainY = trainProvider.getInputProvider(1)

    const testProvider = new deeplearn.InCPUMemoryShuffledInputProviderBuilder([dataset.testX, dataset.testY])
    dataset.testX = testProvider.getInputProvider(0)
    dataset.testY = testProvider.getInputProvider(1)

    buildModel(dataset)
})

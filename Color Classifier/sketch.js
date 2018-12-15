let data;
let model;
let xs, ys;
let rSlider, gSlider, bSlider;
let p;
let message;
let red, green, blue;
let allLabels = [
  "red-ish",
  "green-ish",
  "blue-ish",
  "orange-ish",
  "yellow-ish",
  "pink-ish",
  "purple-ish",
  "brown-ish",
  "grey-ish",
];

function preload() {
  data = loadJSON('colorData.json');
}

function setup() {
  createCanvas(200, 200);
  p = createP("");
  message = createP("");
  red = createP("Red: ");
  rSlider = createSlider(0, 255, 100);
  green = createP("Green: ");
  gSlider = createSlider(0, 255, 100);
  blue = createP("Blue: ");
  bSlider = createSlider(0, 255, 100);
  let colors = [];
  let labels = [];
  for (let record of data.entries) {
    let input = [record.r / 255, record.g / 255, record.b / 255];
    colors.push(input);
    labels.push(allLabels.indexOf(record.label));
  }
  xs = tf.tensor2d(colors);
  let labelsTensor = tf.tensor1d(labels, 'int32');
  ys = tf.oneHot(labelsTensor, 9);

  // creating the model
  model = tf.sequential();
  let hidden = tf.layers.dense({
    units: 12,
    activation: 'sigmoid',
    inputDim: 3
  });
  let output = tf.layers.dense({
    units: 9,
    activation: 'softmax'
  });
  model.add(hidden);
  model.add(output);

  //create an optimizer
  const optimizer = tf.train.sgd(0.2);

  // compile the model
  // "meanSqauredError" --> "categoricalCrossEntropy"
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy'
  });

  // train the model
  train();
}

async function train() {
  let options = {
    epochs: 15,
    validationSplit: 0.1,
    shuffle: true,
    callbacks: {
      onTrainBegin: () => {
        message.html("Training...Please wait!");
        message.style("color", "red");
      },
      onTrainEnd: () => {
        message.html("Training Completed!");
        message.style("color", "green");
      },
      onBatchEnd: () => {
        return tf.nextFrame()
      }
    }
  }
  return await model.fit(xs, ys, options);
}

function draw() {
  let r = rSlider.value();
  let g = gSlider.value();
  let b = bSlider.value();
  background(r, g, b);
  tf.tidy(() => {
    const xs = tf.tensor2d([
      [r / 255, g / 255, b / 255]
    ])
    let results = model.predict(xs);
    let guess = results.argMax(1).dataSync()[0];
    let label = allLabels[guess];
    p.html("Guess: " + label);
    red.html("Red: " + r);
    green.html("Green: " + g);
    blue.html("Blue: " + b);
  })
}
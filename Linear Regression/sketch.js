let xs = [];
let ys = [];
let m, b;
const LR = 0.1;
let optimizer;

function setup() {
  createCanvas(600, 600);
  optimizer = tf.train.sgd(LR);
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

function predict(xs) {
  const inputs = tf.tensor1d(xs);
  const outputs = inputs.mul(m).add(b);
  return outputs;
}

function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

function mousePressed() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);
  xs.push(x);
  ys.push(y);
}

function tensorize(vals) {
  return tf.tensor1d(vals);
}

function draw() {
  background(0);
  tf.tidy(() => {
    if (xs.length > 0) {
      optimizer.minimize(() => loss(predict(xs), tensorize(ys)));
      for (let i = 0; i < xs.length; i++) {
        let px = map(xs[i], 0, 1, 0, width);
        let py = map(ys[i], 1, 0, 0, height);
        strokeWeight(8);
        stroke(255);
        point(px, py);
      }

      let inputs = [0, 1];
      let outputs = predict(inputs);
      let l = outputs.dataSync();
      let x1 = map(inputs[0], 0, 1, 0, width);
      let x2 = map(inputs[1], 0, 1, 0, width);
      let y1 = map(l[0], 0, 1, height, 0);
      let y2 = map(l[1], 0, 1, height, 0);
      stroke(255);
      strokeWeight(4);
      line(x1, y1, x2, y2);
    }
  });
}
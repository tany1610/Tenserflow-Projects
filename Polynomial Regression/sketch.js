let xs = [];
let ys = [];
let a, b, c;
const LR = 0.1;
let optimizer;

function setup() {
  createCanvas(600, 600);
  optimizer = tf.train.sgd(LR);
  a = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  c = tf.variable(tf.scalar(random(-1, 1)));
}

function predict(xs) {
  const inputs = tf.tensor1d(xs);
  const outputs = inputs.square().mul(a).add(inputs.mul(b)).add(c);
  return outputs;
}

function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

function mousePressed() {
  let x = map(mouseX, 0, width, -1, 1);
  let y = map(mouseY, 0, height, 1, -1);
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
        let px = map(xs[i], -1, 1, 0, width);
        let py = map(ys[i], 1, -1, 0, height);
        strokeWeight(8);
        stroke(255);
        point(px, py);
      }

      let inputs = [];
      for (let x = -1; x < 1; x += 0.05) {
        inputs.push(x);
      }
      let outputs = predict(inputs);
      let l = outputs.dataSync();
      beginShape();
      noFill();
      stroke(255);
      strokeWeight(4);
      for (let i = 0; i < inputs.length; i++) {
        let x = map(inputs[i], -1, 1, 0, width);
        let y = map(l[i], -1, 1, height, 0);
        vertex(x, y);
      }
      endShape();
    }
  });
}
require('dotenv').config();
const express = require('express');
const session = require('express-session');
const bodyParser = require('body-parser');
const cassandra = require('cassandra-driver');
const path = require('path');
const axios = require('axios');

const app = express();

const client = new cassandra.Client({
  cloud: {
    secureConnectBundle: path.join(__dirname, 'secure-connect-mooc.zip') 
  },
  credentials: {
    username: 'token',
    password: process.env.ASTRA_DB_APPLICATION_TOKEN
  },
  keyspace: process.env.ASTRA_DB_KEYSPACE
});

// Connection check
client.connect()
  .then(() => console.log('Đã kết nối AstraDB'))
  .catch(err => console.error('Lỗi kết nối:', err));

// Middleware
app.use(session({
  secret: process.env.SESSION_SECRET,
  resave: false,
  saveUninitialized: true
}));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('public'));
app.set('view engine', 'ejs');

// Routes
app.get('/', (req, res) => {
  if (!req.session.userId) {
    return res.redirect('/login');
  }
  res.redirect('/recommendations');
});

app.get('/login', (req, res) => {
  res.render('login');
});

app.post('/login', async (req, res) => {
  const { userId } = req.body;
  
// Kiểm tra user có tồn tại không
  req.session.userId = userId;
  res.redirect('/recommendations');
});

app.get('/logout', (req, res) => {
  req.session.destroy();
  res.redirect('/login');
});

app.get('/recommendations', async (req, res) => {
  if (!req.session.userId) {
    return res.redirect('/login');
  }

  try {
    // Lấy top 10 khóa học đề xuất
    const recommendationsQuery = `
      SELECT course_id, score 
      FROM recommendations 
      WHERE user_id = ? 
      ORDER BY score DESC 
      LIMIT 10`;
    const recommendationsResult = await client.execute(recommendationsQuery, [req.session.userId], { prepare: true });

    // Lấy thông tin chi tiết các khóa học đề xuất
    const recommendedCourses = [];
    for (const row of recommendationsResult.rows) {
      const courseQuery = 'SELECT * FROM courses WHERE course_id = ?';
      const courseResult = await client.execute(courseQuery, [row.course_id], { prepare: true });
      recommendedCourses.push({
        ...courseResult.rows[0],
        score: row.score
      });
    }
    // Lấy tất cả khóa học
    const allCoursesQuery = 'SELECT * FROM courses';
    const allCoursesResult = await client.execute(allCoursesQuery);

    res.render('recommendations', {
      userId: req.session.userId,
      recommendedCourses,
      allCourses: allCoursesResult.rows
    });
  } catch (err) {
    console.error(err);
    res.status(500).send('Server error');
  }
});

app.post('/enroll', async (req, res) => {
  if (!req.session.userId) {
    return res.redirect('/login');
  }

  const { courseId } = req.body;

  try {
    // Code bạn làm ở đây đừng có xóa, chạy được á.
    // const query = 'INSERT INTO user_course (user_id, course_id, enroll_time) VALUES (?, ?, ?)';
    // const params = [req.session.userId, courseId, new Date()];
    // await client.execute(query, params, { prepare: true });

    // await axios.post('http://localhost:8000/produce', {
    //   user_id: req.session.userId,
    //   course_id: courseId
    // }, {
    //   headers: {
    //     'Content-Type': 'application/json'
    //   }
    // });
    res.redirect('/recommendations');
  } catch (err) {
    console.error(err);
    res.status(500).send('Server error');
  }
});

app.get('/products/:courseId', async (req, res) => {
  const { courseId } = req.params;
  
  try {
    const courseQuery = 'SELECT * FROM course WHERE course_id = ?';
    const courseResult = await client.execute(courseQuery, [courseId], { prepare: true });

    if (courseResult.rows.length === 0) {
      return res.status(404).send('Product not found');
    }

    const course = courseResult.rows[0];

    const recRes = await axios.get(`http://localhost:8000/recommendations/${req.session.userId}`);
    const recommendedCourseIds = recRes.data.recommendedCourses || [];

    const recommendedCourses = [];
    for (const id of recommendedCourseIds) {
      const courseQueryRec = 'SELECT * FROM courses WHERE course_id = ?';
      const recResult = await client.execute(courseQueryRec, [id], { prepare: true });
      if (recResult.rows.length > 0) {
        recommendedCourses.push(recResult.rows[0]);
      }
    }

    res.render('course_detail', {
      course,                   
      recommendedCourses,       
      userId: req.session.userId
    });
  } catch (err) {
    console.error(err);
    res.status(500).send('Server error');
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
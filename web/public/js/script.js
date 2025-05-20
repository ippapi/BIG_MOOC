function openTab(evt, tabId) {
    const tabContents = document.querySelectorAll('.tabcontent, .tab-content');
    tabContents.forEach(tab => tab.style.display = 'none');

    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => btn.classList.remove('active'));

    document.getElementById(tabId).style.display = 'block';

    evt.currentTarget.classList.add('active');
}


app.post('/enroll', async (req, res) => {
    if (!req.session.userId) {
      return res.redirect('/login');
    }
    const { courseId } = req.body;
    try {
      res.redirect(`/course/${courseId}`);
    } catch (err) {
      console.error(err);
      res.status(500).send('Server error');
    }
  });
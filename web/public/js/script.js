function openTab(evt, tabName) {
    const tabcontent = document.getElementsByClassName("tab-content");
    for (let i = 0; i < tabcontent.length; i++) {
        tabcontent[i].classList.remove("active");
    }

    const tablinks = document.getElementsByClassName("tab-btn");
    for (let i = 0; i < tablinks.length; i++) {
        tablinks[i].classList.remove("active");
    }

    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
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
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ClassLibrary;

namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
        XyLy xyLy;
        private void button1_Click(object sender, EventArgs e)
        {

            //Bitmap bitmap = (Bitmap)pictureBox1.Image;

            //pictureBox2.Image = 

            string file = null;
            OpenFileDialog ofdImages = new OpenFileDialog();
            PictureBox objpt = new PictureBox();
            if (ofdImages.ShowDialog() == DialogResult.OK)
            {
                file = ofdImages.FileName;
            }

            Bitmap bitmap = new Bitmap(file);

            pictureBox2.Image = bitmap;

            string xc = xyLy.XuLyDuLieu(bitmap);

            //string xc = xyLy.test2();
            MessageBox.Show(xc);

            //String cxk = "";
            //for(int i = 0; i <5; i++)
            //{
            //    char cx = '8';
            //    cxk = cxk + cx;
            //}

            //MessageBox.Show(cxk);



        }

        private void Form1_Load(object sender, EventArgs e)
        {
            xyLy = new XyLy();
            bool sc = xyLy.Load_Cascade();
        }
    }
}

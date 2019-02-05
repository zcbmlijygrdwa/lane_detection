//inspired by https://medium.com/@mithi/advanced-lane-finding-using-computer-vision-techniques-7f3230b6c6f2


#include <opencv2/opencv.hpp>
#include "../cvminecraft/curve_detection/CurveDetection.hpp"

using namespace cv;

int main(int argc, char** argv)
{

    const char* default_file = "../../cvminecraft/data/image/lane1.jpg";
    const char* filename = argc >=2 ? argv[1] : default_file;

    //Input and Output Image;
    Mat input;

    //Load the image
    input = imread(filename, 1 );


    imshow("Input",input);
    waitKey(0);



    //-------------------------- Perspective transform into top view -----------------
    Mat topview_c;

    // Input Quadilateral or Image plane coordinates
    Point2f inputQuad[4];
    // Output Quadilateral or World plane coordinates
    Point2f outputQuad[4];

    // Lambda Matrix
    Mat lambda( 2, 4, CV_32FC1 );

    // Set the lambda matrix the same type and size as input
    lambda = Mat::zeros( input.rows, input.cols, input.type() );

    // The 4 points that select quadilateral on the input , from top-left in clockwise order
    // These four pts are the sides of the rect box used as input 

    inputQuad[0] = Point2f(622,430);
    inputQuad[1] = Point2f(662,430);
    inputQuad[2] = Point2f(1276,621);
    inputQuad[3] = Point2f(3, 621);
    // The 4 points where the mapping is to be done , from top-left in clockwise order
    outputQuad[0] = Point2f( 0,0 );
    outputQuad[1] = Point2f( input.cols-1,0);
    outputQuad[2] = Point2f( input.cols-1,input.rows-1);
    outputQuad[3] = Point2f( 0,input.rows-1  );

    // Get the Perspective Transform Matrix i.e. lambda 
    lambda = getPerspectiveTransform( inputQuad, outputQuad );

    cout<<"lambda = "<<endl<<lambda<<endl;

    // Apply the Perspective Transform just found to the src
    warpPerspective(input,topview_c,lambda,topview_c.size() );


    imshow("topview_c",topview_c);
    waitKey(0);


    //-------------------------- HSV color filter to remove useless colors -----------------
    Mat topview_yellow_filter;


    const int max_value = 255;

    //int low_H = 0;
    //int high_H = 180;

    //int low_S = 120;
    //int high_S = 255;

    //int low_L = 40;
    //int high_L = 255;

    int low_H = 0;
    int high_H = 180;

    int low_S = 70;
    int high_S = 255;

    int low_V = 170;
    int high_V = 255;


    //p = { 'sat_thresh': 120, 'light_thresh': 40, 'light_thresh_agr': 205,\n"

    // Convert from BGR to HSV colorspace
    cvtColor(topview_c, topview_yellow_filter, COLOR_RGB2HSV);


    //self.color_cond1 = (self.s > self.sat_thresh) & (self.l > self.light_thresh)
    //self.color_cond2 = self.l > self.light_thresh_agr
    //b = self.z.copy()
    //b[(self.color_cond1 | self.color_cond2)] = 1

    Mat color_cond1, color_cond2;

    cout<<"low_H = "<<low_H<<endl;

    // Detect the object based on HSV Range Values
    inRange(topview_yellow_filter, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), topview_yellow_filter);

    imshow("topview_yellow_filter",topview_yellow_filter);
    waitKey(0);

    //-------------------------- CurveDetection -----------------

    CurveDetection cd;

    Mat topview_yellow_curve;
    topview_yellow_curve = topview_c.clone();

    cd.setBinaryInput(topview_yellow_filter);
    cd.findLocations();
    cd.solve();
    vector<Point2d> result = cd.getResult();
    int trow = 0;
    int tcol = 0;

    Mat loc_mat = Mat::zeros( topview_yellow_filter.rows, topview_yellow_filter.cols, topview_yellow_filter.type() );

    vector<Point> loc = cd.getLocations();

    for(int i = 0; i<result.size(); i++)
    {
        //cout<<"result[i] = "<<result[i]<<endl;
        trow = result[i].y;
        tcol = result[i].x;
        //cout<<"tcol = "<<tcol<<", trow = "<<trow<<" ,input.cols = "<<input.cols<<", input.rows = "<<input.rows<<endl;
        //input.at<uchar>(trow,tcol) = 255;
        topview_yellow_curve.at<Vec3b>(trow,tcol) = Vec3b(255,0,0);
        //output.at<uchar>(trow,tcol) = 255;
    }

    for(int i = 0; i<loc.size(); i++)
    {
        trow = loc[i].y;
        tcol = loc[i].x;
        loc_mat.at<uchar>(trow,tcol) = 255;
    }

    //Display input and output
    imshow("loc_mat",loc_mat);
    imshow("topview_yellow_curve",topview_yellow_curve);
    waitKey(0);

    ////-------------------------- LineDetection -----------------

    //Mat line_vis1 = topview_c.clone();

    //// Standard Hough Line Transform
    //vector<Vec2f> lines; // will hold the results of the detection
    //HoughLines(topview_yellow_filter, lines, 5, 5*CV_PI/180, 250, 0, 0 ); // runs the actual detection
    //// Draw the lines
    //for( size_t i = 0; i < lines.size(); i++ )
    //{
    //    float rho = lines[i][0], theta = lines[i][1];
    //    Point pt1, pt2;
    //    double a = cos(theta), b = sin(theta);
    //    double x0 = a*rho, y0 = b*rho;
    //    pt1.x = cvRound(x0 + 1000*(-b));
    //    pt1.y = cvRound(y0 + 1000*(a));
    //    pt2.x = cvRound(x0 - 1000*(-b));
    //    pt2.y = cvRound(y0 - 1000*(a));
    //    line( line_vis1, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
    //}

    ////// Probabilistic Line Transform
    ////vector<Vec4i> linesP; // will hold the results of the detection
    ////HoughLinesP(dst, linesP, 1, CV_PI/180, 50, 50, 10 ); // runs the actual detection
    ////// Draw the lines
    ////for( size_t i = 0; i < linesP.size(); i++ )
    ////{
    ////    Vec4i l = linesP[i];
    ////    line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
    ////}

    //imshow("line_vis1",line_vis1);
    //waitKey(0);

    //-------------------------- Project curve points back -----------------
    Mat lambda_inv = lambda.inv();
    cout<<"lambda_inv = "<<endl<<lambda_inv<<endl;
    Mat chs[2];


    Mat draw_points = Mat(result);
    Mat draw2;
    split(draw_points,chs);//split source 

    hconcat(chs[0],chs[1],draw2); 
    hconcat(draw2,Mat::ones(Size(1,draw2.rows),draw2.type()),draw2); 
    //hconcat(draw_points,Mat::ones(draw_points.rows,1,draw_points.type()),draw2); 
    draw2 = draw2.t();


    Mat bp = lambda_inv*draw2;
    bp = bp.t();

    cout<<"bp.rows = "<<bp.rows<<endl;

    for(int i = 0 ; i < bp.rows;i++)
    {
        //cout<<"bp.at<64F>(i,2) = "<<bp.at<double>(i,2)<<endl;
        bp.row(i) = bp.row(i)/bp.at<double>(i,2);
    }





    Mat re_proj = input.clone();



    for(int i = 0; i<bp.rows-1; i++)
    {
        line(re_proj, Point(bp.at<double>(i,0),bp.at<double>(i,1)), Point(bp.at<double>(i+1,0),bp.at<double>(i+1,1)), Scalar(255,0,0), 5, 8, 0);
        //cout<<"result[i] = "<<result[i]<<endl;
        trow = bp.at<double>(i,1);
        tcol = bp.at<double>(i,0);
        //cout<<"tcol = "<<tcol<<", trow = "<<trow<<" ,input.cols = "<<input.cols<<", input.rows = "<<input.rows<<endl;
        //input.at<uchar>(trow,tcol) = 255;
        re_proj.at<Vec3b>(trow,tcol) = Vec3b(255,0,0);
        //output.at<uchar>(trow,tcol) = 255;
    }
    imshow("re_proj",re_proj);
    waitKey(0);

    return 0;
}

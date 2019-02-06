//inspired by https://medium.com/@mithi/advanced-lane-finding-using-computer-vision-techniques-7f3230b6c6f2


#include <opencv2/opencv.hpp>
#include "../cvminecraft/curve_detection/CurveDetection.hpp"

using namespace cv;


VideoCapture cap;

int main(int argc, char** argv)
{

    cap = VideoCapture("../data/project_video.mp4");

    int trow = 0;
    int tcol = 0;

    const char* default_file = "../../cvminecraft/data/image/lane1.jpg";
    const char* filename = argc >=2 ? argv[1] : default_file;

    //Input and Output Image;
    Mat input;

    //Load the image
    input = imread(filename, 1 );

    int frameCount = 0;

    while(cap.isOpened())
    {
        cap>>input;


        //imshow("Input",input);
        //waitKey(0);


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

        inputQuad[0] = Point2f(540,475);
        inputQuad[1] = Point2f(770,475);
        inputQuad[2] = Point2f(1260,676);
        inputQuad[3] = Point2f(100, 676);


        //inputQuad[0] = Point2f(622,430);
        //inputQuad[1] = Point2f(662,430);
        //inputQuad[2] = Point2f(1276,621);
        //inputQuad[3] = Point2f(3, 621);
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


        //imshow("topview_c",topview_c);
        //waitKey(0);

        //imwrite("outputs/top_view/topview"+to_string(frameCount)+".jpg",topview_c);


        //-------------------------- HSV color filter to remove useless colors -----------------
        Mat topview_yellow_filter;
        Mat topview_white_filter;


        const int max_value = 255;

        int low_H = 0;
        int high_H = 180;

        int low_S = 100;
        int high_S = 230;

        int low_V = 220;
        int high_V = 255;

        // Convert from BGR to HSV colorspace
        cvtColor(topview_c, topview_yellow_filter, COLOR_RGB2HSV);

        cout<<"low_H = "<<low_H<<endl;

        // Detect the object based on HSV Range Values
        inRange(topview_yellow_filter, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), topview_yellow_filter);


        //configure for white color
        low_H = 0;
        high_H = 180;

        low_S = 0;
        high_S = 20;

        low_V = 220;
        high_V = 255;

        // Convert from BGR to HSV colorspace
        cvtColor(topview_c, topview_white_filter, COLOR_RGB2HSV);

        cout<<"low_H = "<<low_H<<endl;

        // Detect the object based on HSV Range Values
        inRange(topview_white_filter, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), topview_white_filter);

        //imshow("topview_yellow_filter",topview_yellow_filter);
        //imshow("topview_white_filter",topview_white_filter);
        //waitKey(0);

        //-------------------------- CurveDetection -----------------


        Mat topview_yellow_curve;
        Mat topview_white_curve;
        Mat topview_curve;

        topview_yellow_curve = topview_c.clone();
        topview_white_curve = topview_c.clone();
        topview_curve = topview_c.clone();

        CurveDetection cd_y;
        cd_y.setBinaryInput(topview_yellow_filter);
        cd_y.findLocations();
        cd_y.solve();
        vector<Point2d> result_y = cd_y.getResult();

        Mat loc_mat_y = Mat::zeros( topview_yellow_filter.rows, topview_yellow_filter.cols, topview_yellow_filter.type() );
        vector<Point> print_points_y = cd_y.getLocations();
        for(uint i = 0; i<result_y.size(); i++)
        {
            trow = result_y[i].y;
            tcol = result_y[i].x;
            //topview_yellow_curve.at<Vec3b>(trow,tcol) = Vec3b(255,0,0);
            topview_curve.at<Vec3b>(trow,tcol) = Vec3b(255,0,0);
        }

        for(uint i = 0; i<print_points_y.size(); i++)
        {
            trow = print_points_y[i].y;
            tcol = print_points_y[i].x;
            loc_mat_y.at<uchar>(trow,tcol) = 255;
        }

        CurveDetection cd_w;
        cd_w.setBinaryInput(topview_white_filter);
        cd_w.findLocations();
        cd_w.solve();
        vector<Point2d> result_w = cd_w.getResult();

        Mat loc_mat_w = Mat::zeros( topview_white_filter.rows, topview_white_filter.cols, topview_white_filter.type() );
        vector<Point> print_points_w = cd_w.getLocations();
        for(uint i = 0; i<result_w.size(); i++)
        {
            trow = result_w[i].y;
            tcol = result_w[i].x;
            //topview_white_curve.at<Vec3b>(trow,tcol) = Vec3b(255,0,0);
            topview_curve.at<Vec3b>(trow,tcol) = Vec3b(0,255,0);
        }

        for(uint i = 0; i<print_points_w.size(); i++)
        {
            trow = print_points_w[i].y;
            tcol = print_points_w[i].x;
            loc_mat_w.at<uchar>(trow,tcol) = 255;
        }

        ////Display input and output
        //imshow("loc_mat_w",loc_mat_w);
        //imshow("loc_mat_y",loc_mat_y);
        //imshow("topview_white_curve",topview_white_curve);
        //imshow("topview_yellow_curve",topview_yellow_curve);
        //imshow("topview_curve",topview_curve);
        //waitKey(0);

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
        Mat re_proj = input.clone();
        Mat lambda_inv = lambda.inv();
        cout<<"lambda_inv = "<<endl<<lambda_inv<<endl;
        Mat chs[2];


        Mat draw_points_y = Mat(result_y);
        Mat draw2;
        split(draw_points_y,chs);//split source 

        hconcat(chs[0],chs[1],draw2); 
        hconcat(draw2,Mat::ones(Size(1,draw2.rows),draw2.type()),draw2); 
        draw2 = draw2.t();

        Mat bp = lambda_inv*draw2;
        bp = bp.t();

        cout<<"bp.rows = "<<bp.rows<<endl;

        for(uint i = 0 ; i < bp.rows;i++)
        {
            //cout<<"bp.at<64F>(i,2) = "<<bp.at<double>(i,2)<<endl;
            bp.row(i) = bp.row(i)/bp.at<double>(i,2);
        }

        vector<Point> back_proj_points_y;
        for(uint i = 0; i<bp.rows-1; i++)
        {
            back_proj_points_y.push_back(Point(bp.at<double>(i,0),bp.at<double>(i,1)));
            //line(re_proj, Point(bp.at<double>(i,0),bp.at<double>(i,1)), Point(bp.at<double>(i+1,0),bp.at<double>(i+1,1)), Scalar(255,0,0), 5, 8, 0);
            //cout<<"result[i] = "<<result[i]<<endl;
            trow = bp.at<double>(i,1);
            tcol = bp.at<double>(i,0);
            //re_proj.at<Vec3b>(trow,tcol) = Vec3b(255,0,0);
        }


        Mat draw_points_w = Mat(result_w);
        split(draw_points_w,chs);//split source 

        hconcat(chs[0],chs[1],draw2); 
        hconcat(draw2,Mat::ones(Size(1,draw2.rows),draw2.type()),draw2); 
        draw2 = draw2.t();

        bp = lambda_inv*draw2;
        bp = bp.t();

        cout<<"bp.rows = "<<bp.rows<<endl;

        for(uint i = 0 ; i < bp.rows;i++)
        {
            //cout<<"bp.at<64F>(i,2) = "<<bp.at<double>(i,2)<<endl;
            bp.row(i) = bp.row(i)/bp.at<double>(i,2);
        }

        vector<Point> back_proj_points_w;
        for(uint i = 0; i<bp.rows-1; i++)
        {
            back_proj_points_w.push_back(Point(bp.at<double>(i,0),bp.at<double>(i,1)));
            //line(re_proj, Point(bp.at<double>(i,0),bp.at<double>(i,1)), Point(bp.at<double>(i+1,0),bp.at<double>(i+1,1)), Scalar(0,255,0), 5, 8, 0);
            trow = bp.at<double>(i,1);
            tcol = bp.at<double>(i,0);
            //re_proj.at<Vec3b>(trow,tcol) = Vec3b(0,255,0);
        }

        Point back_proj_points2[1][back_proj_points_y.size()+back_proj_points_w.size()];
        
        for(uint i = 0 ; i <back_proj_points_y.size();i++)
        {
            back_proj_points2[0][i] = back_proj_points_y[i];
        }

        for(uint i = 0 ; i <back_proj_points_w.size();i++)
        {
            back_proj_points2[0][back_proj_points_y.size()+i] = back_proj_points_w[back_proj_points_w.size()-1-i];
        }

        const Point* ppt[1] = {back_proj_points2[0]};
        int npt[] = {back_proj_points_y.size()+back_proj_points_w.size()};
        //polylines(re_proj, ppt, npt, 1, 1, Scalar(255),1,8,0);
        Mat paint_cover = Mat::zeros(Size(re_proj.cols,re_proj.rows),re_proj.type());
        fillPoly(paint_cover, ppt, npt, 1, Scalar(255,0,120,0.5));

        double alpha = 0.9;
        cv::addWeighted(re_proj, 1.0, paint_cover, 0.5 , 0.0, re_proj); 

        //fillConvexPoly(re_proj,back_proj_points,Scalar(0,0,255),16,0);

        imshow("re_proj",re_proj);
        //imwrite("outputs/re_proj/re_proj"+to_string(frameCount)+".jpg",re_proj);
        waitKey(1);
        frameCount++;
    }
    return 0;
}

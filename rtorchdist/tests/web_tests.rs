use actix_web::{http::StatusCode, test, App};
use rtorchdist::routes::index;

#[actix_rt::test]
async fn test_index() {
    let mut app = test::init_service(App::new().service(index)).await;
    let req = test::TestRequest::get().uri("/").to_request();
    let resp = test::call_service(&mut app, req).await;

    assert_eq!(resp.status(), StatusCode::OK);

    let response_body = test::read_body(resp).await;
    let expected_body = "Send an image payload using curl with the following command:\ncurl -X POST -H \"Content-Type: multipart/form-data\" -F \"image=@/path/to/your/image.jpg\" http://127.0.0.1:8080/predict";
    assert_eq!(response_body, expected_body);
}

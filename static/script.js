function searchMedical() {
    let query = document.getElementById("query-input").value.trim();
    if (query === "") {
        alert("Vui lòng nhập nội dung cần tìm!");
        return;
    }

    fetch("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        let resultDiv = document.getElementById("result");
        if (data.error) {
            resultDiv.innerHTML = `<p style="color: red;">${data.error}</p>`;
        } else {
            if (data["Loại"] === "Thuốc") {
                resultDiv.innerHTML = `
                    <h3>${data["Tên"]} (Thuốc)</h3>
                    <p><strong>Công dụng:</strong> ${data["Công dụng"]}</p>
                    <p><strong>Tác dụng phụ:</strong> ${data["Tác dụng phụ"]}</p>
                    <p><strong>Liều lượng:</strong> ${data["Liều lượng"]}</p>
                `;
            } else if (data["Loại"] === "Bệnh") {
                resultDiv.innerHTML = `
                    <h3>${data["Tên"]} (Bệnh)</h3>
                    <p><strong>Triệu chứng:</strong> ${data["Triệu chứng"]}</p>
                    <p><strong>Nguyên nhân:</strong> ${data["Nguyên nhân"]}</p>
                    <p><strong>Điều trị:</strong> ${data["Điều trị"]}</p>
                `;
            }
        }
    })
    .catch(error => console.error("Lỗi:", error));
}

//function searchDrug() {
//    let drugName = document.getElementById("drug-input").value.trim();
//    if (drugName === "") {
//        alert("Vui lòng nhập tên thuốc!");
//        return;
//    }
//
//    fetch("/search", {
//        method: "POST",
//        headers: { "Content-Type": "application/json" },
//        body: JSON.stringify({ drug_name: drugName })
//    })
//    .then(response => response.json())
//    .then(data => {
//        let resultDiv = document.getElementById("result");
//        if (data.error) {
//            resultDiv.innerHTML = `<p style="color: red;">${data.error}</p>`;
//        } else {
//            resultDiv.innerHTML = `
//                <h3>${data["Tên thuốc"]}</h3>
//                <p><strong>Công dụng:</strong> ${data["Công dụng"]}</p>
//                <p><strong>Tác dụng phụ:</strong> ${data["Tác dụng phụ"]}</p>
//                <p><strong>Liều lượng:</strong> ${data["Liều lượng"]}</p>
//            `;
//        }
//    })
//    .catch(error => console.error("Lỗi:", error));
//}

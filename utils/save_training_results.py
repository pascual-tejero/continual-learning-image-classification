import xlsxwriter
import os

def save_training_results(dicc_results, workbook, id_task, training_name="naive"):
    """
    Create an excel file to save the results of the experiments
    
    """

    name_worksheet = f"{training_name}_after_task{str(id_task)}"
    # print(name_worksheet)

    worksheet = workbook.add_worksheet(name_worksheet) # Create a worksheet

    # Merge cells A1:D1 and E1:H1
    merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'fg_color': '#D7E4BC'})
    
    worksheet.merge_range('A1:E1', 'Global metrics', merge_format)
    worksheet.merge_range('G1:K1', 'Test task metrics', merge_format)    

    # Create the headers
    worksheet.write(1, 0, "Task")
    worksheet.write(1, 1, "Epoch")
    worksheet.write(1, 2, "Train loss")
    worksheet.write(1, 3, "Val loss")
    worksheet.write(1, 4, "Test average accuracy")

    worksheet.write(1, 6, "Task")
    worksheet.write(1, 7, "Epoch")
    worksheet.write(1, 8, "Test loss")
    worksheet.write(1, 9, "Test accuracy")

    # Write the results in the excel file    
    row = 2
    col = 0
    for i in range(len(dicc_results["Train task"])):
        worksheet.write(row, col, dicc_results["Train task"][i])
        worksheet.write(row, col+1, dicc_results["Train epoch"][i])
        worksheet.write(row, col+2, dicc_results["Train loss"][i])
        worksheet.write(row, col+3, dicc_results["Val loss"][i])
        worksheet.write(row, col+4, dicc_results["Test average accuracy"][i])
        row += 1

    row = 2
    col = 0
    count = 1
    for i in range(len(dicc_results["Test task"])):
        for j in range(len(dicc_results["Test task"][i])):
            worksheet.write(row, col+6, dicc_results["Test task"][i][j])
            worksheet.write(row, col+7, count)
            worksheet.write(row, col+8, dicc_results["Test loss"][i][j])
            worksheet.write(row, col+9, dicc_results["Test accuracy"][i][j])
            row += 1
        count += 1
    


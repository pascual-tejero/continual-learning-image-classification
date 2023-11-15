import xlsxwriter
import os

def save_training_results(dicc_results, workbook, task, training_name="naive"):
    """
    Create an excel file to save the results of the experiments
    
    """

    name_worksheet = f"{training_name}_after_task{str(task+1)}"
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
    worksheet.merge_range('G1:J1', 'Test task metrics', merge_format)    

    # Create the headers
    worksheet.write(1, 0, "Task")
    worksheet.write(1, 1, "Epoch")
    worksheet.write(1, 2, "Train loss")
    worksheet.write(1, 3, "Val loss")
    worksheet.write(1, 4, "Test average accuracy")

    worksheet.write(1, 5, "Task")
    worksheet.write(1, 6, "Epoch")
    worksheet.write(1, 7, "Test loss")
    worksheet.write(1, 8, "Test accuracy")



    # Write the results (example below)
    # {
    #     'Train task': [[1, 1, 1], [1, 1, 1], [1, 1, 1]], 
    #     'Train epoch': [[1, 2, 3], [1, 2, 3], [1, 2, 3]], 
    #     'Train loss': [[0.06199685484170914, 7.287984772119671e-05, 0.002928048139438033], [0.06199685484170914, 7.287984772119671e-05, 0.002928048139438033], [0.06199685484170914, 7.287984772119671e-05, 0.002928048139438033]], 
    #     'Test task': [[1, 2], [1, 2], [1, 2]], 
    #     'Test loss': [[0.0025341288498252875, 1.9125297585030931], [0.001379269548828799, 2.4220029647710404], [0.0015170017510350662, 2.6597196647079784]], 
    #     'Test accuracy': [[99.15023793337865, 0.0], [99.4901427600272, 0.0], [99.55812372535691, 0.0]], 
    #     'Test average accuracy': [[49.575118966689324, 49.7450713800136, 49.779061862678454], [49.575118966689324, 49.7450713800136, 49.779061862678454], [49.575118966689324, 49.7450713800136, 49.779061862678454]]
    #  }
    
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
            worksheet.write(row, col+5, dicc_results["Test task"][i][j])
            worksheet.write(row, col+6, count)
            worksheet.write(row, col+7, dicc_results["Test loss"][i][j])
            worksheet.write(row, col+8, dicc_results["Test accuracy"][i][j])
            row += 1
        count += 1
    


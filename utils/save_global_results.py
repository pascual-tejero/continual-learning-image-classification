import os
import xlsxwriter


def save_global_results(dicc_avg_acc, args):
    """
    Create an excel file to save the results of the experiments
    
    """
    # Path to save the results
    if args.dataset == "mnist":
        path_file = f"./results/mnist/global_results.xlsx"
    elif args.dataset == "cifar10":
        path_file = f"./results/cifar10/global_results.xlsx"

    if os.path.exists(path_file):  # If the file exists
        os.remove(path_file)  # Remove the file if it exists

    # Create the excel file
    if os.path.exists(path_file):  # If the file exists
        os.remove(path_file)  # Remove the file if it exists

    workbook = xlsxwriter.Workbook(path_file)  # Create the excel file
    worksheet = workbook.add_worksheet(f"Global metrics ")  # Create a worksheet

    merge_format = workbook.add_format({
    'bold': 1,
    'border': 1,
    'align': 'center',
    'valign': 'vcenter',
    'fg_color': '#D7E4BC'})

    worksheet.merge_range('A1:G1', f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.lr}", merge_format)


    # worksheet.write(1, 0, "Training method")
    # worksheet.write(1, 1, "Test average accuracy after task 1")
    # worksheet.write(1, 2, "Test average accuracy after task 2")

    # row = 2
    # col = 0
    # for key, value in dicc_avg_acc.items():
    #     worksheet.write(row, col, key)
    #     worksheet.write(row, col+1, value[0])
    #     worksheet.write(row, col+2, value[1])
    #     row += 1

    # workbook.close()  # Close the workbook

    
    # Create the headers
    worksheet.write(1, 0, "Test task")
    worksheet.write(2, 0, "Test average accuracy after task 1")
    worksheet.write(3, 0, "Test average accuracy after task 2")

    # Write the results in the excel file
    row = 1
    col = 1
    for key, value in dicc_avg_acc.items():
        worksheet.write(row, col, key)
        row += 1
        worksheet.write(row, col, value[0])
        row += 1
        worksheet.write(row, col, value[1])
        row = 1
        col += 1

    workbook.close()  # Close the workbook